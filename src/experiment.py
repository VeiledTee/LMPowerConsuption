import gc
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import Dataset, load_dataset
from tqdm import tqdm

from config import CONFIG
from inference import inference, load_model_and_tokenizer
from prompts import build_prompt
from retrieval import load_wiki, retrieve_hotpot
from scorers import exact_match, f1_score
from utils import (convert_seconds, count_bools, ensure_config_dirs,
                   setup_logging)

# Supress ollama http logs
logging.getLogger("httpx").setLevel(logging.WARNING)
# Supress codecarbon warnings
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logger = setup_logging()
ensure_config_dirs()


def run() -> None:
    """Main experiment runner."""
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    start_time = time.time()

    logger.info(f"Starting experiment with config:\n{CONFIG}")
    logger.info(f"Using device: {CONFIG.device}")

    try:
        dataset = load_config_dataset(data_dir)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    for model_name, provider in CONFIG.model_types.items():
        model_start_time = time.time()
        logger.info(f"\n{'=' * 60}\nRunning model: {model_name}\n{'=' * 60}")

        tokenizer, model = None, None
        if not CONFIG.retrieval_only and provider != "ollama":
            tokenizer, model = load_model_safely(model_name)
        elif not CONFIG.retrieval_only and provider == "ollama":
            tokenizer = True
            model = model_name

        model_modes = CONFIG.modes.get(model_name, {})
        logger.info(f"\n{'+' * 30}\nRunning modes: {model_modes}\n{'+' * 30}")
        for mode_tag, include_passage in model_modes.items():
            run_model_mode(
                model_name,
                mode_tag,
                include_passage,
                dataset,
                tokenizer,
                model,
                provider,
            )

        cleanup_resources(model, tokenizer)

        model_time = time.time() - model_start_time
        logger.info(f"Completed {model_name} in {format_time(model_time)}")

    total_time = time.time() - start_time
    logger.info(
        f"\n{'=' * 60}\nExperiment completed in {format_time(total_time)}\n{'=' * 60}"
    )


def load_config_dataset(data_dir: Path) -> Dataset:
    """Load dataset based on configuration."""
    dataset_path = data_dir / CONFIG.dataset_file

    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data)

    if "hotpot" in CONFIG.dataset_name:
        return load_dataset(
            CONFIG.dataset_name,
            CONFIG.config,
            split=CONFIG.split,
            trust_remote_code=True,
        )

    if "boolq" in CONFIG.dataset_name:
        return load_dataset(
            CONFIG.dataset_name,
            split=CONFIG.split,
            trust_remote_code=True,
        )

    raise ValueError(f"Unsupported dataset: {CONFIG.dataset_name}")


def load_model_safely(model_name: str):
    """Load model with error handling."""
    try:
        tokenizer, model = load_model_and_tokenizer(model_name)
        logger.info(f"Loaded model: {model_name}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None, None


def cleanup_resources(model, tokenizer) -> None:
    """Release model resources and clean memory."""
    if model:
        del model
    if tokenizer:
        del tokenizer
    if CONFIG.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def format_time(seconds: float) -> str:
    """Convert seconds to human-readable format."""
    h, m, s = convert_seconds(seconds)
    return f"{h}h {m}m {s}s"


def run_model_mode(
    model_name: str,
    mode_tag: str,
    include_passage: bool,
    dataset: Dataset,
    tokenizer: any,
    model: any,
    provider: str,
) -> None:
    """Run evaluation for a specific model and mode."""
    dataset_id = "boolq" if "boolq" in CONFIG.dataset_name else "hotpot"
    csv_path = (
        CONFIG.result_dir
        / f"{dataset_id}{'_128' if 'mini' in CONFIG.dataset_file else ''}_{model_name.split('/')[-1].replace(':', '-')}_{mode_tag}.csv"
    )

    wiki_data = load_wikipedia_if_needed(mode_tag)
    start_idx = get_resume_index(csv_path)
    results = []

    pbar = tqdm(
        total=len(dataset) - start_idx, desc=f"{model_name} ({mode_tag})", unit="sample"
    )

    for idx in range(start_idx, len(dataset)):
        try:
            result = process_current_sample(
                idx,
                dataset[idx],
                mode_tag,
                include_passage,
                wiki_data,
                model,
                tokenizer,
                model_name,
                provider,
            )
            if result:
                results.append(result)
                manage_batch_saving(results, csv_path)
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
        pbar.update(1)

    save_results(results, csv_path)
    pbar.close()
    logger.info(f"Completed {mode_tag} mode for {model_name}")


def load_wikipedia_if_needed(mode_tag: str) -> tuple | None:
    """Load Wikipedia data for retrieval mode."""
    if mode_tag != "q+r":
        return None

    try:
        t0 = time.time()
        wiki_data = load_wiki()
        logger.info(f"Loaded Wikipedia corpus and indexes in {convert_seconds(time.time() - t0)}")
        return wiki_data
    except Exception as e:
        logger.error(f"Wikipedia loading failed: {str(e)}")
        return None


def get_resume_index(csv_path: Path) -> int:
    """Get resume index from existing results file."""
    if not csv_path.exists():
        return 0

    try:
        existing_df = pd.read_csv(csv_path)
        return 0 if existing_df.empty else int(existing_df["qid"].max()) + 1
    except Exception as e:
        logger.warning(f"Couldn't read existing CSV: {str(e)}")
        return 0


def process_current_sample(
    idx: int,
    sample: dict,
    mode_tag: str,
    include_passage: bool,
    wiki_data: tuple | None,
    model: any,
    tokenizer: any,
    model_name: str,
    provider: str,
) -> dict | None:
    """Process a single sample and return results."""
    # Initialize metrics
    retrieval_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }

    # Build and process prompt
    sample_for_prompt = sample.copy()

    # Retrieve context if needed
    if mode_tag == "q+r":
        retrieved_context, retrieval_metrics = retrieve_context(
            sample, wiki_data
        )  # Rename variable
        sample_for_prompt["retrieved_context"] = retrieved_context  # New key

    prompt = build_prompt(sample_for_prompt, include_passage)

    # Generate prediction
    prediction, inference_metrics = generate_prediction(
        prompt, model, tokenizer, model_name, mode_tag, provider
    )

    # Process BoolQ special case
    if "boolq" in CONFIG.dataset_name:
        prediction, inference_metrics = process_boolq_prediction(
            prediction, model_name, inference_metrics
        )

    # Prepare gold answer
    gold_answer = sample["answer"]
    if not isinstance(gold_answer, str):
        gold_answer = str(gold_answer)

    # Calculate scores
    em = exact_match(prediction, gold_answer)
    f1 = f1_score(prediction, gold_answer)

    return {
        "qid": idx,
        "pred": prediction,
        "gold": gold_answer,
        "em": em,
        "f1": f1,
        "inference_duration (s)": inference_metrics["duration"],
        "inference_energy_consumed (kWh)": inference_metrics["energy_consumed"],
        "inference_emissions (kg)": inference_metrics["emissions"],
        "retrieval_duration (s)": retrieval_metrics["duration"],
        "retrieval_energy_consumed (kWh)": retrieval_metrics["energy_consumed"],
        "retrieval_emissions (kg)": retrieval_metrics["emissions"],
    }


def retrieve_context(sample: dict, wiki_data: tuple | None) -> tuple[str, dict]:
    """Retrieve context passage based on dataset type."""
    retrieval_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }
    context = ""

    if not wiki_data:
        return context, retrieval_metrics

    if "hotpot" in CONFIG.dataset_name:
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_hotpot(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)
        # Extract context from sample
        context = " ".join(
            sent for section in sample["context"]["sentences"] for sent in section
        )
    elif "boolq" in CONFIG.dataset_name:
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_hotpot(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)
        context = sample.get("passage", "")

    return context, retrieval_metrics


def generate_prediction(
    prompt: str,
    model: any,
    tokenizer: any,
    model_name: str,
    mode_tag: str,
    provider: str,
) -> tuple[str, dict]:
    """Generate model prediction with metrics."""
    inference_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }
    prediction = ""

    if model and tokenizer:
        full_output, i_metrics = inference(
            prompt, model, tokenizer, model_name, mode_tag, provider
        )
        prediction = full_output.split("Answer: ")[-1].strip()
        inference_metrics.update(i_metrics)
    return prediction, inference_metrics


def process_boolq_prediction(
    prediction: str, model_name: str, inference_metrics: dict
) -> tuple[str, dict]:
    """Process BoolQ prediction and measure emissions."""
    with EmissionsTracker(
        project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_simplifying",
        log_level="error",
    ) as tracker:
        processed_pred = count_bools(prediction)

    new_metrics = {
        "duration": float(tracker.final_emissions_data.duration),
        "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
        "emissions": float(tracker.final_emissions_data.emissions),
    }
    result = {}
    for d in [inference_metrics, new_metrics]:
        for k, v in d.items():
            result[k] = result.get(k, 0) + v

    return processed_pred, result


def manage_batch_saving(results: list[dict], csv_path: Path) -> None:
    """Save results if batch size reached and clear resources."""
    if len(results) < CONFIG.batch_size:
        return

    save_results(results, csv_path)
    results.clear()

    if CONFIG.device == "cuda":
        torch.cuda.empty_cache()


def save_results(results: list[dict], csv_path: Path) -> None:
    """Save results to CSV file."""
    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
    except Exception as e:
        logger.exception(f"Critical error: {str(e)}")
        raise

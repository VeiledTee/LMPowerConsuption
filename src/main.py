import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from config import CONFIG
from inference import inference, load_model_and_tokenizer
from prompts import build_prompt
from retrieval import load_wiki, retrieve
from scorers import exact_match, f1_score
from utils import convert_seconds, ensure_config_dirs, setup_logging

logger = setup_logging()
ensure_config_dirs()


def run() -> None:
    """Main experiment runner."""
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    start_time = time.time()
    logger.info(f"Starting experiment with config:\n{CONFIG}")
    logger.info(f"Using device: {CONFIG.device}")

    # dataset_path = Path(data_dir / "hotpot_mini_128.jsonl")
    dataset_path = Path(data_dir / "boolq_mini_128.jsonl")
    # dataset_path = Path(data_dir / "hotpot_mini_129.jsonl")

    try:
        if dataset_path.exists():
            logger.info(f"Loading dataset from {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list(data)
            logger.info(
                f"Loaded mini dataset with {len(dataset)} samples from {dataset_path}"
            )
        elif "hotpot" in CONFIG.dataset_name:
            dataset: Dataset = load_dataset(
                CONFIG.dataset_name,
                CONFIG.config,
                split=CONFIG.split,
                trust_remote_code=True,
            )
            logger.info(f"Loaded Hotpot dataset with {len(dataset)} samples")
        elif "boolq" in CONFIG.dataset_name:
            dataset: Dataset = load_dataset(
                CONFIG.dataset_name,
                split=CONFIG.split,
                trust_remote_code=True,
            )
            logger.info(f"Loaded BoolQ dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    for model_name in CONFIG.model_candidates:
        model_start = time.time()
        logger.info(f"\n{'=' * 60}\nRunning model: {model_name}\n{'=' * 60}")

        tokenizer, model = None, None
        if not CONFIG.retrieval_only:
            try:
                tokenizer, model = load_model_and_tokenizer(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                continue

        for mode_tag, include_passage in CONFIG.modes.items():
            run_mode(model_name, mode_tag, include_passage, dataset, tokenizer, model)

        if model:
            del model
            if tokenizer:
                del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        model_time = time.time() - model_start
        h, m, s = convert_seconds(model_time)
        logger.info(f"Completed {model_name} in {h}h {m}m {s}s")

    total_time = time.time() - start_time
    h, m, s = convert_seconds(total_time)
    logger.info(f"\n{'=' * 60}\nExperiment completed in {h}h {m}m {s}s\n{'=' * 60}")


def run_mode(
    model_name: str,
    mode_tag: str,
    include_passage: bool,
    dataset: Dataset,
    tokenizer: any,
    model: any,
) -> None:
    """Run evaluation for a specific model and mode."""
    # Determine dataset identifier for CSV naming
    dataset_id = "boolq" if "boolq" in CONFIG.dataset_name else "hotpot"
    csv_path: Path = (
        CONFIG.result_dir / f"{dataset_id}_{model_name.split('/')[-1]}_{mode_tag}.csv"
    )
    # csv_path: Path = CONFIG.result_dir / f"{dataset_id}_128_{model_name.split('/')[-1]}_{mode_tag}.csv"
    # csv_path: Path = CONFIG.result_dir / f"{dataset_id}_512_{model_name.split('/')[-1]}_{mode_tag}.csv"

    # Load Wikipedia only for HotpotQA in retrieval mode
    wiki_data: tuple | None = None
    if mode_tag == "q+r" and "hotpot" in CONFIG.dataset_name:
        try:
            wiki_data = load_wiki()
            logger.info("Loaded Wikipedia corpus and indexes")
        except Exception as e:
            logger.error(f"Wikipedia loading failed: {str(e)}")
            return

    # Resume from last saved index
    start_idx: int = 0
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                start_idx = int(existing_df["qid"].max()) + 1
                logger.info(f"Resuming from sample {start_idx}")
        except Exception as e:
            logger.warning(f"Couldn't read existing CSV: {str(e)}")

    results: list[dict] = []
    pbar = tqdm(
        total=len(dataset) - start_idx, desc=f"{model_name} ({mode_tag})", unit="sample"
    )

    for idx in range(start_idx, len(dataset)):
        try:
            sample = dataset[idx]
            sample_id = sample.get("id", idx)

            # Initialize metrics
            retrieval_metrics = {
                "duration": 0.0,
                "energy_consumed": 0.0,
                "emissions": 0.0,
            }
            context = ""

            context = ""
            if mode_tag == "q+r":
                if "hotpot" in CONFIG.dataset_name and wiki_data:
                    # HotpotQA: Retrieve passages
                    docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
                    context, ret_metrics = retrieve(
                        sample["question"], vectorizer, tfidf_matrix, titles, inv_index
                    )
                    retrieval_metrics.update(ret_metrics)
                elif "boolq" in CONFIG.dataset_name:
                    # BoolQ: Use provided passage
                    context = sample.get("passage", "")

            # Build prompt with context
            sample_for_prompt = sample.copy()
            if context:
                sample_for_prompt["context"] = context
            prompt = build_prompt(sample_for_prompt, include_passage)

            # Generate prediction
            prediction: str = ""
            inference_metrics: dict = {
                "duration": 0.0,
                "energy_consumed": 0.0,
                "emissions": 0.0,
            }
            if model and tokenizer:
                full_output, inference_metrics = inference(
                    prompt, model, tokenizer, model_name, mode_tag
                )
                prediction = full_output.split("Answer: ")[-1].strip()

            # Convert gold answer to string for BoolQ
            gold_answer = sample["answer"]
            if not isinstance(gold_answer, str):
                gold_answer = str(gold_answer)

            # Calculate scores
            em: float = exact_match(prediction, gold_answer)
            f1: float = f1_score(prediction, gold_answer)

            # Collect results
            results.append(
                {
                    "qid": idx,
                    "pred": prediction,
                    "gold": gold_answer,
                    "em": em,
                    "f1": f1,
                    "inference_duration (s)": inference_metrics["duration"],
                    "inference_energy_consumed (kWh)": inference_metrics[
                        "energy_consumed"
                    ],
                    "inference_emissions (kg)": inference_metrics["emissions"],
                    "retrieval_duration (s)": retrieval_metrics["duration"],
                    "retrieval_energy_consumed (kWh)": retrieval_metrics[
                        "energy_consumed"
                    ],
                    "retrieval_emissions (kg)": retrieval_metrics["emissions"],
                }
            )

            if len(results) >= CONFIG.batch_size:
                save_results(results, csv_path)
                results = []
                if CONFIG.device == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
        pbar.update(1)

    if results:
        save_results(results, csv_path)
    pbar.close()
    logger.info(f"Completed {mode_tag} mode for {model_name}")


def save_results(results: list[dict], csv_path: Path) -> None:
    """
    Save results incrementally to a CSV file.

    Args:
        results: List of result dictionaries to append.
        csv_path: Path to the output CSV file.
    """
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

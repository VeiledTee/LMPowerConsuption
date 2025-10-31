import gc
import json
import logging
import os
import smtplib
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.config import CONFIG
from src.inference import inference, load_model_and_tokenizer
from src.prompts import build_prompt
from src.retrieval import load_HotpotQA_wiki, load_2WikiMultiHopQA_wiki, retrieve_2wikimultihop, retrieve_hotpot
from src.squad_scorers import compute_exact, compute_f1
from src.utils import (convert_seconds, count_bools, ensure_config_dirs,
                   extract_2wiki_gold_context, setup_logging)

# Supress ollama http logs
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
# Supress codecarbon warnings
logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

logger = setup_logging()
ensure_config_dirs()

CONCURRENCY = 8
PROMPT_BUFFER = []
RESULT_BUFFER = []


def run(file_suffix: None | str = "") -> None:
    """Main experiment runner."""
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    start_time = time.time()

    logger.info(f"Starting experiment with config:\n{CONFIG}")
    logger.info(f"Using device: {CONFIG.device}")

    # Load dataset
    try:
        dataset = load_config_dataset(data_dir)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    # Preload Wikipedia if needed
    wiki_data = None
    if any("q+r" in modes for modes in CONFIG.modes.values()):
        logger.info(f"Retrieval mode requested - loading wiki")
        t0 = time.time()
        if "2wikimultihop" in CONFIG.dataset_name.lower():
            logger.info("Loading mappings for 2WikiMultiHopQA context retrieval...")
            wiki_data = load_2WikiMultiHopQA_wiki()
        else:
            logger.info("Loading mappings for HotpotQA context retrieval...")
            wiki_data = load_HotpotQA_wiki()

        h, m, s = convert_seconds(time.time() - t0)
        logger.info(f"Loaded wiki in {h}h{m}m{s}s")

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
                wiki_data,
                file_suffix,
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

    if '2wiki' in str(dataset_path).lower():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # flatten or stringify non-scalar fields
        for d in data:
            d["context"] = json.dumps(d["context"])
            d["supporting_facts"] = json.dumps(d["supporting_facts"])
            d["evidences"] = json.dumps(d["evidences"])

        return Dataset.from_dict({k: [ex[k] for ex in data] for k in data[0]})

    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data)

    if "hotpot" in CONFIG.dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name,
            CONFIG.config,
            split=CONFIG.split,
            trust_remote_code=True,
        )
        return dataset

    elif "boolq" in CONFIG.dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name,
            split=CONFIG.split,
            trust_remote_code=True,
        )
        return dataset

    elif "squad" in CONFIG.dataset_name:
        dataset = load_dataset(
            CONFIG.dataset_name,
            split=CONFIG.split,
        )

        n = CONFIG.dataset_size
        subset = dataset.shuffle(seed=CONFIG.seed).select(range(n))

        return subset

    raise ValueError(f"Unsupported dataset: {CONFIG.dataset_name}")


def load_model_safely(model_name: str):
    """Load model with fallback to CPU on CUDA OOM."""
    try:
        tokenizer, model = load_model_and_tokenizer(model_name)
        logger.info(f"Loaded model: {model_name}")
        return tokenizer, model
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"CUDA OOM — retrying on CPU for model: {model_name}")
            torch.cuda.empty_cache()
            CONFIG.device = "cpu"  # switch device in config
            try:
                tokenizer, model = load_model_and_tokenizer(model_name)
                return tokenizer, model
            except Exception as inner_e:
                logger.error(f"Fallback to CPU failed: {str(inner_e)}")
        else:
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
    dataset,
    tokenizer: any,
    model: any,
    provider: str,
    wiki_data: tuple,
    file_suffix: None | str = "",
) -> None:
    if "hotpot" in CONFIG.dataset_name:
        from scorers_hotpot import exact_match, f1_score  # use Hotpot-style
    elif '2wiki' in CONFIG.dataset_name.lower():
        from scorers_2wiki import exact_match, f1_score  # use 2Wiki-style
    else:
        from scorers import exact_match, f1_score  # use general BoolQ-style

    # Run evaluation for a specific model and mode, with concurrent inference and batched writes
    dataset_id = CONFIG.dataset_name.split(r'/')[-1]
    csv_path = (
        CONFIG.result_dir
        / f"{dataset_id}_{model_name.split('/')[-1].replace(':', '-')}_{mode_tag}"
        f"{'_128' if 'mini' in CONFIG.dataset_file else ''}"
        f"{'_dev' if 'dev' in CONFIG.dataset_file else ''}"
        f"{'_think' if CONFIG.think == True else ''}"
        f"{'_' + file_suffix if file_suffix != '' else ''}.csv"
    )
    logger.info(f"Saving results to: {csv_path}")
    start_idx = get_resume_index(csv_path)
    overall_t0 = time.time()

    # Pre-build prompts + metadata
    prompt_t0 = time.time()
    jobs = []
    total_ret = {"duration": 0.0, "energy_consumed": 0.0, "emissions": 0.0}
    for idx in range(start_idx, len(dataset)):
        sample = dataset[idx]
        ret_metrics = None
        if mode_tag == "q+r":
            ctx, ret_metrics = retrieve_context(sample, wiki_data)
            sample = {**sample, "retrieved_context": ctx}

            if ret_metrics:
                for k in total_ret:
                    total_ret[k] += ret_metrics.get(k, 0.0)
        prompt = build_prompt(sample, include_passage)
        jobs.append((idx, sample, prompt, ret_metrics))

    logger.info(
        f"Built {len(jobs)} prompts in {format_time(time.time() - prompt_t0)} "
        f"(+ retrieval: {format_time(total_ret['duration'])}, "
        f"{total_ret['energy_consumed']:.4f} kWh, {total_ret['emissions']:.4f} kg)"
    )

    result_buffer = []

    if provider == "ollama":
        _ = inference("Ready?", model_name, mode_tag, provider)

        for idx, sample, prompt, ret_metrics in tqdm(
            jobs, desc=f"{model_name} ({mode_tag})", total=len(jobs)
        ):
            full_output, inf_metrics = inference(prompt, model_name, mode_tag, provider)

            pred = extract_prediction(full_output)

            if "squad" in CONFIG.dataset_name:
                # For SQuAD, we need to handle the answer format
                gold_answers = sample["answers"]["text"] if "answers" in sample else [""]

                # Calculate EM and F1 using the official SQuAD functions
                em = max(compute_exact(a, pred) for a in gold_answers)
                f1 = max(compute_f1(a, pred) for a in gold_answers)

                row = {
                    "qid": sample["id"],
                    "original_pred": full_output.replace(",", " ").replace("  ", " ").replace("\n", " ").replace(".", '').strip(),
                    "pred": pred,
                    "gold": str(gold_answers),  # Store all possible answers as string
                    "em": em,
                    "f1": f1,
                    "inference_duration (s)": inf_metrics["duration"],
                    "inference_energy_consumed (kWh)": inf_metrics["energy_consumed"],
                    "inference_emissions (kg)": inf_metrics["emissions"],
                    "retrieval_duration (s)": (
                        ret_metrics.get("duration") if ret_metrics else 0.0
                    ),
                    "retrieval_energy_consumed (kWh)": (
                        ret_metrics.get("energy_consumed") if ret_metrics else 0.0
                    ),
                    "retrieval_emissions (kg)": (
                        ret_metrics.get("emissions") if ret_metrics else 0.0
                    ),
                }
            else:

                if "boolq" in CONFIG.dataset_name:
                    pred, inf_metrics = process_boolq_prediction(
                        pred, model_name, inf_metrics
                    )

                em = exact_match(pred, sample["answer"])
                f1 = f1_score(pred, sample["answer"])

                row = {
                    "qid": idx,
                    "original_pred": full_output.replace(",", " ")
                    .replace("  ", " ")
                    .replace("\n", " "),
                    "pred": pred,
                    "gold": sample["answer"],
                    "em": em,
                    "f1": f1,
                    "inference_duration (s)": inf_metrics["duration"],
                    "inference_energy_consumed (kWh)": inf_metrics["energy_consumed"],
                    "inference_emissions (kg)": inf_metrics["emissions"],
                    "retrieval_duration (s)": (
                        ret_metrics.get("duration") if ret_metrics else 0.0
                    ),
                    "retrieval_energy_consumed (kWh)": (
                        ret_metrics.get("energy_consumed") if ret_metrics else 0.0
                    ),
                    "retrieval_emissions (kg)": (
                        ret_metrics.get("emissions") if ret_metrics else 0.0
                    ),
                }

            result_buffer.append(row)

            if len(result_buffer) >= CONFIG.batch_size:
                save_results(result_buffer, csv_path)
                result_buffer.clear()

        if result_buffer:
            save_results(result_buffer, csv_path)

    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_job = {
                executor.submit(
                    generate_prediction,
                    prompt,
                    model,
                    tokenizer,
                    model_name,
                    mode_tag,
                    provider,
                ): (idx, sample, ret_metrics)
                for idx, sample, prompt, ret_metrics in jobs
            }

            for fut in tqdm(
                as_completed(future_to_job),
                total=len(future_to_job),
                desc=f"{model_name} ({mode_tag})",
            ):
                idx, sample, ret_metrics = future_to_job[fut]
                try:
                    pred, inf_metrics = fut.result()

                    if "squad" in CONFIG.dataset_name:
                        # For SQuAD, we need to handle the answer format
                        gold_answers = sample["answers"]["text"] if "answers" in sample else [""]

                        # Calculate EM and F1 using the official SQuAD functions
                        em = max(compute_exact(a, pred) for a in gold_answers)
                        f1 = max(compute_f1(a, pred) for a in gold_answers)

                        row = {
                            "qid": sample["id"],
                            "original_pred": pred.replace(",", " ").replace("  ", " ").replace("\n", " "),
                            "pred": pred,
                            "gold": str(gold_answers),  # Store all possible answers as string
                            "em": em,
                            "f1": f1,
                            "inference_duration (s)": inf_metrics["duration"],
                            "inference_energy_consumed (kWh)": inf_metrics["energy_consumed"],
                            "inference_emissions (kg)": inf_metrics["emissions"],
                            "retrieval_duration (s)": (
                                ret_metrics.get("duration") if ret_metrics else 0.0
                            ),
                            "retrieval_energy_consumed (kWh)": (
                                ret_metrics.get("energy_consumed") if ret_metrics else 0.0
                            ),
                            "retrieval_emissions (kg)": (
                                ret_metrics.get("emissions") if ret_metrics else 0.0
                            ),
                        }
                    else:

                        if "boolq" in CONFIG.dataset_name:
                            pred, inf_metrics = process_boolq_prediction(
                                pred, model_name, inf_metrics
                            )

                        em = exact_match(pred, sample["answer"])
                        f1 = f1_score(pred, sample["answer"])
                        row = {
                            "qid": idx,
                            "original_pred": pred,
                            "pred": pred,
                            "gold": sample["answer"],
                            "em": em,
                            "f1": f1,
                            "inference_duration (s)": inf_metrics["duration"],
                            "inference_energy_consumed (kWh)": inf_metrics[
                                "energy_consumed"
                            ],
                            "inference_emissions (kg)": inf_metrics["emissions"],
                            "retrieval_duration (s)": (
                                ret_metrics.get("duration") if ret_metrics else 0.0
                            ),
                            "retrieval_energy_consumed (kWh)": (
                                ret_metrics.get("energy_consumed") if ret_metrics else 0.0
                            ),
                            "retrieval_emissions (kg)": (
                                ret_metrics.get("emissions") if ret_metrics else 0.0
                            ),
                        }

                    result_buffer.append(row)
                except Exception as e:
                    logger.error(f"Inference failed for idx={idx}: {e}")

                if len(result_buffer) >= CONFIG.batch_size:
                    save_results(result_buffer, csv_path)
                    result_buffer.clear()

    if result_buffer:
        save_results(result_buffer, csv_path)

    logger.info(
        f"Completed {mode_tag} for {model_name} in {format_time(time.time() - overall_t0)}"
    )


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


def retrieve_context(sample: dict, wiki_data: tuple | None) -> tuple[str, dict]:
    """Retrieve context passage based on dataset type."""
    retrieval_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }
    context = ""

    if "hotpot" in CONFIG.dataset_name:
        if not wiki_data:
            return context, retrieval_metrics
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_hotpot(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)
        # Extract context from sample
        if "passage" in sample.keys():
            context = " ".join(
                sent for section in sample["passage"] for sent in section
            )
        else:
            context = " ".join(
                sent for section in sample["context"]["sentences"] for sent in section
            )

    elif "2wikimultihop" in CONFIG.dataset_name.lower():
        if not wiki_data:
            return context, retrieval_metrics
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        print(f"SAMPLE: {sample}")

        # Extract question data for 2WikiMultiHopQA
        question_text = sample["question"]
        question_type = sample.get("type", "comparison")  # comparison, inference, compositional, bridge-comparison
        gold_titles = [sf[0] for sf in sample["supporting_facts"]]

        # Call the new 2WikiMultiHopQA retrieval function
        _, ret_metrics = retrieve_2wikimultihop(
            question=question_text,
            gold_titles=gold_titles,
            question_type=question_type,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            titles=titles,
            docs=docs,
        )

        context = extract_2wiki_gold_context(sample)
        retrieval_metrics.update(ret_metrics)

    elif "boolq" in CONFIG.dataset_name:
        if not wiki_data:
            return context, retrieval_metrics
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_hotpot(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)
        context = sample.get("passage", "")

    elif "squad" in CONFIG.dataset_name:
        if not wiki_data:
            return context, retrieval_metrics
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_hotpot(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)
        # For SQuAD, use the provided context
        context = sample["context"]

    return context, retrieval_metrics


def extract_prediction(full_output: str) -> str:
    # If model uses <think>...True, extract the last line or the final word
    if "<think>" in full_output:
        full_output = full_output.split("</think>")[-1].strip()
    # If model doesn't think and prepends reply with "Answer:" split on that instead
    elif "Answer: " in full_output:
        full_output = full_output.split("Answer:")[-1].strip()
    # Fallback is taking the original output
    return full_output.strip()


def generate_prediction(
    prompt: str,
    model: any,
    tokenizer: any,
    model_name: str,
    mode_tag: str,
    provider: str,
) -> tuple[str, dict]:
    """Generate model prediction with metrics and fallback on CUDA OOM."""
    inference_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }
    full_output = ""

    try:
        if model and tokenizer:
            full_output, i_metrics = inference(
                prompt, model, tokenizer, model_name, mode_tag, provider
            )
            inference_metrics.update(i_metrics)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(
                f"CUDA OOM during inference — retrying on CPU for {model_name}"
            )
            torch.cuda.empty_cache()
            CONFIG.device = "cpu"

            full_output, i_metrics = inference(
                prompt, model, tokenizer, model_name, mode_tag, provider
            )
            inference_metrics.update(i_metrics)
        else:
            logger.error(f"Inference failed: {str(e)}")

    return full_output, inference_metrics


def process_boolq_prediction(
    prediction: str, model_name: str, inference_metrics: dict
) -> tuple[str, dict]:
    """Process BoolQ prediction and measure emissions."""
    with EmissionsTracker(
        save_to_file=False,
        project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_simplifying",
        log_level="error",
    ) as tracker:
        processed_pred = extract_prediction(count_bools(prediction))

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


def save_results(results: list[dict], csv_path: Path) -> None:
    """Save results to CSV file."""
    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def send_email(from_addr: str, to_addr: str, subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = CONFIG.from_email
    smtp_pass = CONFIG.smtp_password
    if not (smtp_user and smtp_pass):
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD must be set")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


if __name__ == "__main__":
    try:
        run()
        send_email(
            from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
            to_addr=CONFIG.to_email,
            subject="Completed experiment.py with no issues.",
            body="No errors",
        )
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
    except Exception as error:
        logger.exception(f"Critical error: {str(error)}")
        send_email(
            from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
            to_addr=CONFIG.to_email,
            subject="Script Error",
            body=f"The script crashed with:\n\n{error}",
        )
        raise

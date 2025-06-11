import bz2
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ─── fixed hyper‑params ────────────────────────────────────────────────
MODEL_NAME = (
    "distilbert/distilgpt2"  # openai-community/gpt2-xl OR distilbert/distilgpt2
)
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
N_SAMPLES = None
MAX_NEW_TOK = 64
BATCH_SIZE = 32
DEVICE = "cpu"
MODES = {"q": False, "q+r": True}  # {"q": False, "q+r": True}
WIKI_DIR = "enwiki-20171001-pages-meta-current-withlinks-processed"
CORPUS_CACHE = Path("wiki.pkl")
TFIDF_CACHE = Path("tfidf.pkl")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
print(f"{'=' * 25}\nMODEL: {MODEL_NAME}\nMODES: {MODES}\n{'=' * 25}")

ENERGY_DIR = Path("Energy").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# MODEL_NAME = "openai-community/gpt2-xl"  # openai-community/gpt2-xl OR distilbert/distilgpt2
# DATASET_NAME = "hotpotqa/hotpot_qa"
# CONFIG = "fullwiki"
# SPLIT = "validation"
# N_SAMPLES = None
# MAX_NEW_TOK = 64
# BATCH_SIZE = 32
# DEVICE = "cpu"
# MODES = {"q": False}  # {"q": False, "q+r": True}
# print(f"{'='*25}\nMODEL: {MODEL_NAME}\nMODES: {MODES}\n{'='*25}")
#
# ENERGY_DIR = Path("Energy").resolve()
# ENERGY_DIR.mkdir(exist_ok=True)

# ─── quiet library chatter ────────────────────────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)


# ─── token normaliser & per‑row scorers ────────────────────────
def convert_seconds(total_seconds):
    total_hours = total_seconds // 3600
    total_seconds %= 3600
    total_minutes = total_seconds // 60
    total_seconds %= 60
    return total_hours, total_minutes, total_seconds


def _normalize(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pt, gt = _normalize(pred).split(), _normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)


def build_prompt(ex: dict, include_passage: bool) -> str:
    q = ex["question"]
    if not include_passage:
        return f"Question: {q}\nAnswer:"
    titles = {t for t in ex["supporting_facts"]["title"]}
    context = ""
    if titles:
        context = ". ".join(ex["context"]["title"])
        for s in ex["context"]["sentences"]:
            context += "".join(s)
    return f"Context: {context}\n\nQuestion: {q}\nAnswer:"


def load_wiki_corpus(base_dir: Path) -> tuple[list[str], list[str]]:
    documents, titles = [], []
    t0 = time.time()
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".bz2"):
                file_path = os.path.join(subdir, file)
                try:
                    with bz2.open(
                        file_path, "rt", encoding="utf-8", errors="ignore"
                    ) as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                if isinstance(obj.get("text"), list) and obj["text"]:
                                    text = " ".join(
                                        "".join(s)
                                        for s in obj["text"]
                                        if isinstance(s, list)
                                    )
                                    if text.strip():
                                        documents.append(text)
                                        titles.append(obj.get("title", ""))
                            except:
                                continue
                except:
                    continue
    total_s = time.time() - t0
    hours, minutes, seconds = convert_seconds(total_s)
    print(
        f"Collected {len(documents)} full articles in {hours} hours, {minutes} minutes, {seconds} seconds"
    )
    return documents, titles


def fit_tfidf(documents: list[str], n_features: int = 2**12):
    vectorizer = HashingVectorizer(
        n_features=n_features, alternate_sign=False, norm="l2", stop_words="english"
    )
    return vectorizer, vectorizer.transform(documents)


def get_corpus_and_index(base_dir: Path):
    if CORPUS_CACHE.exists() and TFIDF_CACHE.exists():
        return joblib.load(CORPUS_CACHE), joblib.load(TFIDF_CACHE)

    docs, titles = load_wiki_corpus(base_dir)
    vec, mat = fit_tfidf(docs)
    joblib.dump((docs, titles), CORPUS_CACHE)
    joblib.dump((vec, mat), TFIDF_CACHE)
    return (docs, titles), (vec, mat)


def retrieve_with_emissions(
    question: str,
    vectorizer,
    tfidf_matrix,
    article_titles: list[str],
    tracker: EmissionsTracker,
    cc_file: Path,
    return_titles: bool = None,
    precomputed_vec=None,
) -> tuple[list[str], dict]:
    """
    Return (top_titles, energy_metrics) while measuring retrieval emissions.

    If `precomputed_vec` (a 1×N sparse row) is provided, it is used
    directly; otherwise the question is transformed on‑the‑fly.
    """
    tracker.start()

    q_vec = (
        precomputed_vec
        if precomputed_vec is not None
        else vectorizer.transform([question])
    )
    scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
    if return_titles:
        top_k_idx = scores.argsort()[-10:][::-1]
        top_titles = [article_titles[i] for i in top_k_idx]
    else:
        top_titles = []

    tracker.stop()

    row = _tail_row(cc_file)
    return top_titles, {
        "duration": float(row["duration"]),
        "emissions": float(row["emissions"]),
        "energy_consumed": float(row["energy_consumed"]),
    }


def _tail_row(path: Path) -> dict:
    """
    Return {duration, emissions, energy_consumed} from the *last line*
    of a CodeCarbon CSV (faster than pandas for a 2‑line file).
    """
    with path.open() as f:
        last = f.readlines()[-1].strip().split(",")

    return {
        "duration": float(last[3]),
        "emissions": float(last[4]),
        "energy_consumed": float(last[12]),
    }


# ─── single‑mode runner (CSV only) ────────────────────────────────────
def run_mode(run_tag: str, include_passage: bool, dataset, model, tokenizer) -> None:
    csv_out = Path(
        f"hotpot_{MODEL_NAME.split('/')[-1]}_{run_tag}_inference_retrieval.csv"
    )

    t0 = time.time()
    (docs, titles), (vectorizer, tfidf_matrix) = get_corpus_and_index(Path(WIKI_DIR))
    total_s = time.time() - t0
    hours, minutes, seconds = convert_seconds(total_s)
    print(
        f"Loaded and indexed {len(docs)} articles in {hours} hours, {minutes} minutes, {seconds} seconds"
    )

    # Resume support
    start_qid = 0
    if csv_out.exists():
        last = pd.read_csv(csv_out)
        if not last.empty:
            start_qid = int(last["qid"].max()) + 1
    print(f"{run_tag}: resuming at qid {start_qid}")

    remaining = len(dataset) - start_qid
    with tqdm(total=remaining, desc=f"Batches {run_tag}", ncols=80) as pbar:
        for batch_start in range(start_qid, len(dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(dataset))
            batch = dataset.skip(batch_start).take(batch_end - batch_start)

            cc_inference_outfile = f"inference_energy_{MODEL_NAME.split('/')[-1]}_{run_tag}_{batch_start}_{batch_end - 1}.csv"
            cc_retrieval_outfile = (
                f"retrieval_energy_tfidf_{run_tag}_{batch_start}_{batch_end - 1}.csv"
            )

            retrieval_tracker = EmissionsTracker(
                project_name=f"hotpot_tfidf_{run_tag}",
                output_dir=str(ENERGY_DIR),
                output_file=cc_retrieval_outfile,
                log_level="error",
            )

            inference_tracker = EmissionsTracker(
                project_name=f"hotpot_{MODEL_NAME.split('/')[-1]}_{run_tag}",
                output_dir=str(ENERGY_DIR),
                output_file=cc_inference_outfile,
                log_level="error",
            )

            batch_results = []
            batch_questions = [ex["question"] for ex in batch]
            batch_q_vecs = vectorizer.transform(batch_questions)

            for local_i, (qid, ex) in enumerate(
                zip(range(batch_start, batch_end), batch)
            ):
                q_vec = batch_q_vecs[local_i]
                _, retrieval_metrics = retrieve_with_emissions(
                    ex["question"],  # still needed for top‑titles list
                    vectorizer,
                    tfidf_matrix,
                    titles,
                    retrieval_tracker,
                    ENERGY_DIR / cc_retrieval_outfile,
                    return_titles=False,
                    precomputed_vec=q_vec,  # add an optional arg in helper
                )

                prompt = build_prompt(ex, include_passage)
                inference_tracker.start()
                with torch.inference_mode():
                    inputs = tokenizer(prompt, return_tensors="pt")
                    try:
                        out = model.generate(
                            **inputs.to(DEVICE),
                            max_new_tokens=MAX_NEW_TOK,
                            do_sample=False,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.5,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    except torch.OutOfMemoryError:
                        out = model.generate(
                            **inputs.to("cpu"),
                            max_new_tokens=MAX_NEW_TOK,
                            do_sample=False,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.5,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                inference_tracker.stop()

                inference_row = _tail_row(ENERGY_DIR / cc_inference_outfile)
                inference_metrics = {
                    "duration": float(inference_row["duration"]),
                    "energy_consumed": float(inference_row["energy_consumed"]),
                    "emissions": float(inference_row["emissions"]),
                }

                pred = (
                    tokenizer.decode(out[0], skip_special_tokens=True)
                    .strip()
                    .split("Answer: ")[-1]
                )
                gold = ex["answer"]
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)

                batch_results.append(
                    {
                        "qid": qid,
                        "pred": pred,
                        "gold": gold,
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

            # Convert batch results to dataframe and save incrementally
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(csv_out, mode="a", header=not csv_out.exists(), index=False)

            pbar.update(len(batch))

    print(f"{run_tag}: finished; results saved to {csv_out}")


# ─── main: load once, run modes ───────────────────────────────────────
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
        )
        .to(DEVICE)
        .eval()
    )
    data = load_dataset(DATASET_NAME, CONFIG, split=SPLIT, trust_remote_code=True)
    if N_SAMPLES:
        data = data.select(range(N_SAMPLES))

    for tag, ctx in MODES.items():
        run_mode(tag, ctx, data, mdl, tok)

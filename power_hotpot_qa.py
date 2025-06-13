import bz2
import gc
import json
import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_log

# ─── fixed hyper‑params ────────────────────────────────────────────────
MODEL_CANDIDATES = [
    # "distilbert/distilgpt2",
    # "openai-community/gpt2-xl",
    "google/gemma-2b-it",  # 2-b Gemma (instruct)
    "google/gemma-7b-it",  # 7-b Gemma (instruct)
    "meta-llama/Llama-2-7b-hf",  # Llama-2 7 b
    "meta-llama/Llama-2-13b-hf",  # Llama-2 13 b
]
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
N_SAMPLES = None
MAX_NEW_TOK = 64
BATCH_SIZE = 16
DEVICE = "cpu"
MODES = {"q": False}  # {"q": False, "q+r": True}
WIKI_DIR = r"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\enwiki-20171001-pages-meta-current-withlinks-processed"
CORPUS_CACHE = Path("wiki.pkl")
TFIDF_CACHE = Path("tfidf.pkl")
INDEX_CACHE = Path("index.pkl")
INTRO_MIN_CHARS = 51
HASH_BITS = 20
TOKEN_PATTERN = r"(?u)\b\w+\b"
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_LINK_RE = re.compile(r"</?a[^>]*>", re.I)

ENERGY_DIR = Path("Energy").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# ─── quiet library chatter ────────────────────────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)


# ─── load hugging facce model ─────────────────────────────────────────
def load_model(name: str, device: str = DEVICE):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = (
        AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,  # Gemma + Llama-2 need this
        )
        .to(device)
        .eval()
    )
    return tok, mdl


# ─── token normaliser & per‑row scorers ────────────────────────
def convert_seconds(total_seconds):
    total_hours = total_seconds // 3600
    total_seconds %= 3600
    total_minutes = total_seconds // 60
    total_seconds %= 60
    return total_hours, total_minutes, total_seconds


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


def title_sentence_pairs(item: dict) -> list[str]:
    pairs = []
    for title, sent_list in zip(item["title"], item["sentences"]):
        line = f"{title}: {' '.join(s.strip() for s in sent_list)}"
        pairs.append(line)
    return pairs


def build_prompt(ex: dict, include_passage: bool) -> str:
    q = ex["question"]
    if include_passage:
        lines = []
        context = ""
        for title, sent_list in zip(ex["context"]["title"], ex["context"]["sentences"]):
            # join the sentence fragments for this title
            paragraph = " ".join(s.strip() for s in sent_list)
            lines.append(f"{title}: {paragraph}")
            # join every title-paragraph pair with a newline
            context += "\n".join(lines)

        return f"Answer the following to the best of your ability. " \
               f"You must provide an answer. " \
               f"If you are unsure, make an educated guess based on what you know and the context provided. " \
               f"Context: {context}\nQuestion: {q}\nAnswer:"
    else:
        return f"Answer the following to the best of your ability. " \
               f"You must provide an answer. " \
               f"If you are unsure, make an educated guess based on what you know. " \
               f"Question: {q}\nAnswer:"


def _normalize(text: str) -> str:
    """Normalize text identically to HotpotQA's preprocessing"""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def strip_links(txt: str) -> str:
    return _LINK_RE.sub("", txt).strip()


def smart_open(p: Path):
    # Hotpot files are *not* compressed; fall back to bz2 only if needed
    return (bz2.open if p.suffix == ".bz2" else open)(
        p, "rt", encoding="utf-8", errors="ignore"
    )


def load_wiki_corpus(root: Path):
    docs, titles = [], []
    for p in root.rglob("wiki_*"):
        print(p)
        if p.is_dir():  # skip sub-dirs
            continue
        with bz2.open(p, "rt") as fh:  # all files in the dump are bz2-compressed
            for line in fh:
                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue

                title = page["title"]
                if page.get("redirect") or title.endswith("(disambiguation)"):
                    continue
                # ------------- pick the first “real” paragraph -------------
                for para_tokens in page["text"]:
                    para = "".join(para_tokens).strip()
                    if len(para) >= INTRO_MIN_CHARS and not para.startswith("=="):
                        para = _LINK_RE.sub("", para)  # strip <a …> tags
                        docs.append(para)
                        titles.append(title)
                        break  # done with this page
            print(len(docs))
    print(f"Documents collected: {len(docs):,}")
    assert len(docs) > 5_000_000, "Corpus too small – wrong path?"
    return docs, titles


def fit_tfidf(docs: list[str]):
    vec = HashingVectorizer(
        n_features=1 << HASH_BITS,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    return vec, vec.transform(docs)


def build_inv_index(docs: list[str]) -> dict[str, list[int]]:
    """Create inverted index matching HotpotQA's tokenization"""
    inv_index = defaultdict(list)
    stop_words = set(ENGLISH_STOP_WORDS)

    for doc_id, doc in enumerate(docs):
        tokens = re.findall(TOKEN_PATTERN, doc)
        # Remove stopwords like HotpotQA's vectorizer
        tokens = [t for t in tokens if t not in stop_words]

        # Add unigrams
        for token in tokens:
            inv_index[token].append(doc_id)

        # Add bigrams (identical to vectorizer's ngram_range=(1,2))
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            inv_index[bigram].append(doc_id)

    return dict(inv_index)


def filter_candidates(q: str, inv: dict, k: int = 5_000) -> list[int]:
    """Filter candidates using HotpotQA's tokenization rules"""
    tokens = re.findall(TOKEN_PATTERN, _normalize(q))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    # Create unigrams and bigrams
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    cnt = Counter()
    for ng in ngrams:
        for doc_id in inv.get(ng, ()):
            cnt[doc_id] += 1

    return [doc for doc, _ in cnt.most_common(k)]


def get_corpus_and_index(base_dir: Path):
    # ─── Corpus ────────────────────────────────────────────────────────────────
    if CORPUS_CACHE.exists():
        print("Wikipedia dump exists")
        docs, titles = joblib.load(CORPUS_CACHE)
    else:
        print("Loading Wikipedia dump…")
        docs, titles = load_wiki_corpus(base_dir)
        joblib.dump((docs, titles), CORPUS_CACHE)

    # ─── TF-IDF ────────────────────────────────────────────────────────────────
    if TFIDF_CACHE.exists():
        print("Fitted Tf-IDF exists")
        vectorizer, tfidf_matrix = joblib.load(TFIDF_CACHE)
    else:
        print("Fitting TF-IDF model…")
        vectorizer, tfidf_matrix = fit_tfidf(docs)
        joblib.dump((vectorizer, tfidf_matrix), TFIDF_CACHE)

    # ─── Inverted Index (depends on TF-IDF) ────────────────────────────────────
    if INDEX_CACHE.exists():
        print("Inverted index exists")
        inv_index = joblib.load(INDEX_CACHE)
    else:
        print("Building inverted index…")
        inv_index = build_inv_index(docs)  # or use vectorizer/tfidf_matrix if required
        joblib.dump(inv_index, INDEX_CACHE)

    return (docs, titles), (vectorizer, tfidf_matrix), inv_index


def retrieve_with_emissions(
    question: str,
    vectorizer,
    tfidf_matrix,
    article_titles: list[str],
    inv_index: dict,
    precomputed_vec=None,
) -> tuple[list[str], dict]:
    """
    Return (top_titles, energy_metrics) while measuring retrieval emissions.

    If `precomputed_vec` (a 1×N sparse row) is provided, it is used
    directly; otherwise the question is transformed on‑the‑fly.
    """
    with EmissionsTracker(save_to_file=False, log_level="error") as tr:
        q_vec = (
            precomputed_vec
            if precomputed_vec is not None
            else vectorizer.transform([question])
        )

        cand_ids = filter_candidates(question, inv_index)  # NEW
        sub_mat = tfidf_matrix[cand_ids]  # slice 5 000×V
        scores = (q_vec @ sub_mat.T).toarray().flatten()

        top_local = scores.argsort()[-10:][::-1]
        top_titles = [article_titles[cand_ids[i]] for i in top_local]

    power_consumption_data = tr.final_emissions_data
    metrics = {
        "duration": float(power_consumption_data.duration),
        "energy_consumed": float(power_consumption_data.energy_consumed),
        "emissions": float(power_consumption_data.emissions),
    }
    return top_titles, metrics


def inference_with_emissions(
    prompt: str,
    model,
    model_name: str,
    tokenizer,
    device: str | torch.device,
    max_new_tokens: int,
    run_tag: str,
) -> tuple[str, dict[str, Any]]:
    """
    Run one inference pass and return (generated_text, metrics).
    Metrics are per-query.

    Parameters
    ----------
    prompt : str
    model  : transformers.PreTrainedModel
    model_name : str representation of model
    tokenizer  : transformers.PreTrainedTokenizer
    device : torch device or "cpu"/"cuda"
    max_new_tokens : int
    run_tag : str

    Returns
    -------
    text : str
    metrics : dict {duration (s), energy_consumed (kWh), emissions (kg)}
    """
    with EmissionsTracker(
        project_name=f"hotpot_{model_name}_{run_tag}",
        log_level="error",
    ) as tr:
        with torch.inference_mode():
            ctx_limit = (
                getattr(model.config, "n_positions", None)
                or getattr(model.config, "max_position_embeddings", None)
                or tokenizer.model_max_length  # final fallback
            )
            max_ctx_len = ctx_limit - max_new_tokens
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_ctx_len,
                padding=False,
            )

            try:
                tokens = model.generate(
                    **inputs.to(device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.5,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except torch.OutOfMemoryError:
                tokens = model.generate(
                    **inputs.to("cpu"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.5,
                    eos_token_id=tokenizer.eos_token_id,
                )

    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    power_consumption_data = tr.final_emissions_data
    metrics = {
        "duration": float(power_consumption_data.duration),
        "energy_consumed": float(power_consumption_data.energy_consumed),
        "emissions": float(power_consumption_data.emissions),
    }
    if torch.cuda.is_available():
        del tokens, inputs  # drop references
        torch.cuda.empty_cache()  # release cached blocks
    gc.collect()
    return text, metrics


def _tail_row(path: Path) -> dict:
    """
    Return {duration, emissions, energy_consumed} from the *last line*
    of a CodeCarbon CSV (faster than pandas for a 2‑line file).
    """
    with path.open() as f:
        last = f.readlines()[-1].strip().split(",")

    return {
        "duration": float(last[4]),
        "emissions": float(last[5]),
        "energy_consumed": float(last[13]),
    }


# ─── single‑mode runner (CSV only) ────────────────────────────────────
def run_mode(
    run_tag: str,
    include_passage: bool,
    dataset,
    hf_model,
    hf_model_name,
    text_tokenizer,
) -> None:
    csv_out = Path(f"hotpot_{hf_model_name.split('/')[-1]}_{run_tag}.csv")

    if run_tag == "q+r":
        t0 = time.time()
        (docs, titles), (vectorizer, tfidf_matrix), inv_index = get_corpus_and_index(
            Path(WIKI_DIR)
        )
        total_s = time.time() - t0
        hours, minutes, seconds = convert_seconds(total_s)
        print(
            f"Loaded and indexed {len(docs)} articles in {hours} hours, {minutes} minutes, {seconds} seconds"
        )
    else:
        print(f"Tag is {run_tag} --- skipping wiki")

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

            batch_results = []
            batch_questions = [ex["question"] for ex in batch]
            if run_tag == "q+r":
                batch_q_vecs = vectorizer.transform(batch_questions)

            for local_i, (qid, ex) in enumerate(
                zip(range(batch_start, batch_end), batch)
            ):
                if run_tag == "q+r":
                    q_vec = batch_q_vecs[local_i]
                    _, retrieval_metrics = retrieve_with_emissions(
                        ex["question"],
                        vectorizer=vectorizer,
                        tfidf_matrix=tfidf_matrix,
                        article_titles=titles,
                        inv_index=inv_index,
                        precomputed_vec=q_vec,
                    )
                else:
                    retrieval_metrics = {
                        "duration": 0,
                        "energy_consumed": 0,
                        "emissions": 0,
                    }
                prompt = build_prompt(ex, include_passage)
                generated_text, inference_metrics = inference_with_emissions(
                    prompt=prompt,
                    model=hf_model,
                    model_name=hf_model_name,
                    tokenizer=text_tokenizer,
                    device=DEVICE,
                    max_new_tokens=MAX_NEW_TOK,
                    run_tag=run_tag,
                )
                pred = generated_text.strip().split("Answer: ")[-1]
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
            del df_batch, batch_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            pbar.update(len(batch))
    del hf_model, text_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"{run_tag}: finished; results saved to {csv_out}")


if __name__ == "__main__":
    # loop through every candidate (comment out to test just one)
    for model_name in MODEL_CANDIDATES:
        print(f"\n=== Running model: {model_name} ===")
        tokenizer, model = load_model(model_name, device=DEVICE)

        data = load_dataset(DATASET_NAME, CONFIG, split=SPLIT, trust_remote_code=True)
        if N_SAMPLES:
            data = data.select(range(N_SAMPLES))

        for tag, ctx in MODES.items():
            run_mode(tag, ctx, data, model, model_name, tokenizer)

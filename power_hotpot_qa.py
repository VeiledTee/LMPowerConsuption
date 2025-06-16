import bz2
import gc
import json
import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_log

@dataclass(frozen=True)
class ExperimentConfig:
    model_candidates: List[str]
    dataset_name: str
    config: str
    split: str
    n_samples: Optional[int]
    max_new_tokens: int
    batch_size: int
    device: str
    modes: Dict[str, bool]
    wiki_dir: Path
    corpus_cache: Path
    tfidf_cache: Path
    index_cache: Path
    intro_min_chars: int
    hash_bits: int
    token_pattern: str
    energy_dir: Path
    retrieval_only: bool
    log_level: str = "INFO"
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "with_context": (
            "Answer the following to the best of your ability. You must provide an answer. "
            "If you are unsure, make an educated guess based on what you know and the context provided. "
            "Context: {context}\nQuestion: {question}\nAnswer:"
        ),
        "without_context": (
            "Answer the following to the best of your ability. You must provide an answer. "
            "If you are unsure, make an educated guess based on what you know. "
            "Question: {question}\nAnswer:"
        )
    })


CONFIG = ExperimentConfig(
    model_candidates=[
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
    ],
    dataset_name="hotpotqa/hotpot_qa",
    config="fullwiki",
    split="validation",
    n_samples=None,
    max_new_tokens=64,
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    modes={"q": False, "q+r": True},
    wiki_dir=Path("data/enwiki-processed"),
    corpus_cache=Path("cache/wiki_v2.pkl"),
    tfidf_cache=Path("cache/tfidf_v2.pkl"),
    index_cache=Path("cache/index_v2.pkl"),
    intro_min_chars=51,
    hash_bits=20,
    token_pattern=r"(?u)\b\w+\b",
    energy_dir=Path("results/energy"),
    retrieval_only=False,
)

# Setup
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.basicConfig(
    level=CONFIG.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.energy_dir / 'experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("energy_eval")
CONFIG.energy_dir.mkdir(parents=True, exist_ok=True)


# Text Normalization
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_links(text: str) -> str:
    return re.sub(r"</?a[^>]*>", "", text).strip()


# Scorers
def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)


def build_prompt(example: Dict[str, Any], include_passage: bool) -> str:
    q = example["question"]
    if include_passage:
        context = "\n".join(
            f"{title}: {' '.join(s.strip() for s in sents)}"
            for title, sents in zip(example["context"]["title"], example["context"]["sentences"])
        )
        return CONFIG.prompt_templates["with_context"].format(context=context, question=q)
    else:
        return CONFIG.prompt_templates["without_context"].format(question=q)


# Utilities
def convert_seconds(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds


def tail_row(path: Path) -> Dict[str, float]:
    with path.open() as f:
        last = f.readlines()[-1].strip().split(",")
    return {
        "duration": float(last[4]),
        "emissions": float(last[5]),
        "energy_consumed": float(last[13]),
    }


def load_wiki(config: ExperimentConfig):
    if config.corpus_cache.exists():
        docs, titles = joblib.load(config.corpus_cache)
    else:
        docs, titles = [], []
        for p in config.wiki_dir.rglob("wiki_*"):
            if p.is_dir():
                continue
            with bz2.open(p, "rt") as fh:
                for line in fh:
                    try:
                        page = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    title = page.get("title", "")
                    if page.get("redirect") or title.endswith("(disambiguation)"):
                        continue
                    for para_tokens in page.get("text", []):
                        para = "".join(para_tokens).strip()
                        if len(para) >= config.intro_min_chars and not para.startswith("=="):
                            para = strip_links(para)
                            docs.append(para)
                            titles.append(title)
                            break
        joblib.dump((docs, titles), config.corpus_cache)

    if config.tfidf_cache.exists():
        vectorizer, tfidf_matrix = joblib.load(config.tfidf_cache)
    else:
        vectorizer = HashingVectorizer(
            n_features=1 << config.hash_bits,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            lowercase=True,
            token_pattern=config.token_pattern,
        )
        tfidf_matrix = vectorizer.transform(docs)
        joblib.dump((vectorizer, tfidf_matrix), config.tfidf_cache)

    if config.index_cache.exists():
        inv_index = joblib.load(config.index_cache)
    else:
        inv_index = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            tokens = re.findall(config.token_pattern, doc)
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            for token in tokens:
                inv_index[token].append(doc_id)
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                inv_index[bigram].append(doc_id)
        inv_index = dict(inv_index)
        joblib.dump(inv_index, config.index_cache)

    return docs, titles, vectorizer, tfidf_matrix, inv_index


def retrieve(question, vectorizer, tfidf_matrix, titles, inv_index, config):
    with EmissionsTracker(save_to_file=False, log_level="error") as tracker:
        q_vec = vectorizer.transform([question])
        tokens = re.findall(config.token_pattern, normalize(question))
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        counter = Counter()
        for ng in ngrams:
            counter.update(inv_index.get(ng, []))
        cand_ids = [doc_id for doc_id, _ in counter.most_common(5000)]
        sub_mat = tfidf_matrix[cand_ids]
        scores = (q_vec @ sub_mat.T).toarray().flatten()
        top_indices = scores.argsort()[-10:][::-1]
        top_titles = [titles[cand_ids[i]] for i in top_indices]
    data = tracker.final_emissions_data
    return top_titles, {
        "duration": float(data.duration),
        "energy_consumed": float(data.energy_consumed),
        "emissions": float(data.emissions),
    }

def inference(prompt, model, tokenizer, config, model_name, run_tag):
    with EmissionsTracker(project_name=f"hotpot_{model_name}_{run_tag}", log_level="error") as tracker:
        with torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - config.max_new_tokens, padding=False).to(config.device)
            tokens = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                eos_token_id=tokenizer.eos_token_id,
            )
    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    data = tracker.final_emissions_data
    return text, {
        "duration": float(data.duration),
        "energy_consumed": float(data.energy_consumed),
        "emissions": float(data.emissions),
    }


def run():
    dataset = load_dataset(CONFIG.dataset_name, CONFIG.config, split=CONFIG.split)
    if CONFIG.n_samples:
        dataset = dataset.select(range(CONFIG.n_samples))

    for model_name in CONFIG.model_candidates:
        logger.info(f"Running model: {model_name}")
        if CONFIG.retrieval_only:
            tokenizer = model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if CONFIG.device == "cuda" else torch.float32,
                device_map="auto" if CONFIG.device == "cuda" else None,
                trust_remote_code=True,
            ).to(CONFIG.device).eval()

        for mode_tag, include_passage in CONFIG.modes.items():
            logger.info(f"Running mode: {mode_tag}")
            csv_path = CONFIG.energy_dir / f"hotpot_{model_name.split('/')[-1]}_{mode_tag}.csv"
            start_idx = 0
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    start_idx = int(df["qid"].max()) + 1

            docs, titles, vectorizer, tfidf_matrix, inv_index = ([], [], None, None, None)
            if mode_tag == "q+r":
                docs, titles, vectorizer, tfidf_matrix, inv_index = load_wiki(CONFIG)

            pbar = tqdm(total=len(dataset) - start_idx, desc=f"{model_name} ({mode_tag})")
            results = []

            for idx in range(start_idx, len(dataset)):
                sample = dataset[idx]
                sample_id = sample.get("id", idx)

                retrieval_metrics = {"duration": 0.0, "energy_consumed": 0.0, "emissions": 0.0}
                if mode_tag == "q+r":
                    _, retrieval_metrics = retrieve(sample["question"], vectorizer, tfidf_matrix, titles, inv_index, CONFIG)

                prompt = build_prompt(sample, include_passage)
                if CONFIG.retrieval_only:
                    prediction = ""
                    inference_metrics = {"duration": 0.0, "energy_consumed": 0.0, "emissions": 0.0}
                else:
                    full_output, inference_metrics = inference(prompt, model, tokenizer, CONFIG, model_name, mode_tag)
                    prediction = full_output.split("Answer: ")[-1].strip()
                em = exact_match(prediction, sample["answer"])
                f1 = f1_score(prediction, sample["answer"])

                results.append({
                    "qid": sample_id,
                    "model": model_name,
                    "mode": mode_tag,
                    "question": sample["question"],
                    "prediction": prediction,
                    "answer": sample["answer"],
                    "em": em,
                    "f1": f1,
                    "retrieval_duration": retrieval_metrics["duration"],
                    "retrieval_energy": retrieval_metrics["energy_consumed"],
                    "retrieval_emissions": retrieval_metrics["emissions"],
                    "inference_duration": inference_metrics["duration"],
                    "inference_energy": inference_metrics["energy_consumed"],
                    "inference_emissions": inference_metrics["emissions"],
                })

                if len(results) >= CONFIG.batch_size or idx == len(dataset) - 1:
                    pd.DataFrame(results).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                    results = []

                if CONFIG.device == "cuda" and idx % CONFIG.batch_size == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                pbar.update(1)
            pbar.close()


if __name__ == "__main__":
    run()

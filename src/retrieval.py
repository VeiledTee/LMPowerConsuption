import bz2
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any

import joblib
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             HashingVectorizer,
                                             TfidfVectorizer)

from src.config import CONFIG
from src.utils import normalize

logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False


def load_wiki() -> tuple[list[str], list[str], HashingVectorizer, Any, dict[str, list[int]]]:
    """
    Load and index a Wikipedia dump for HotpotQA-style retrieval.

    Returns:
        docs (list[str]): list of document texts (first paragraph of each page).
        titles (list[str]): Corresponding page titles.
        vectorizer (HashingVectorizer): Vectorizer used to build TF-IDF features.
        tfidf_matrix (scipy.sparse matrix): Document-term matrix.
        inv_index (dict[str, list[int]]): Inverted index mapping tokens/bigrams to document IDs.
    """
    if CONFIG.corpus_cache.exists():
        docs, titles = joblib.load(CONFIG.corpus_cache)
        if not docs:
            raise ValueError("Corpus cache loaded, but no documents found in it.")
    else:
        docs, titles = [], []
        for p in CONFIG.wiki_dir.rglob("wiki_*"):
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
                        if len(para) >= CONFIG.intro_min_chars and not para.startswith("=="):
                            para = strip_links(para)
                            docs.append(para)
                            titles.append(title)
                            break
        joblib.dump((docs, titles), CONFIG.corpus_cache)

    if CONFIG.tfidf_cache.exists():
        vectorizer, tfidf_matrix = joblib.load(CONFIG.tfidf_cache)
    else:
        vectorizer = HashingVectorizer(
            n_features=1 << CONFIG.hash_bits,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            lowercase=True,
            token_pattern=CONFIG.token_pattern,
        )
        tfidf_matrix = vectorizer.transform(docs)
        joblib.dump((vectorizer, tfidf_matrix), CONFIG.tfidf_cache)

    if CONFIG.index_cache.exists():
        inv_index = joblib.load(CONFIG.index_cache)
    else:
        inv_index = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            tokens = re.findall(CONFIG.token_pattern, doc)
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            for token in tokens:
                inv_index[token].append(doc_id)
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                inv_index[bigram].append(doc_id)
        inv_index = dict(inv_index)
        joblib.dump(inv_index, CONFIG.index_cache)

    return docs, titles, vectorizer, tfidf_matrix, inv_index


def retrieve_hotpot(
    question: str,
    vectorizer: HashingVectorizer,
    tfidf_matrix: Any,
    titles: list[str],
    inv_index: dict[str, list[int]],
) -> tuple[list[str], dict[str, float]]:
    """
    Retrieve top candidate documents for a HotpotQA-style question.

    Args:
        question (str): Input question.
        vectorizer (HashingVectorizer): Pre-built TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): Document-term matrix for all Wikipedia docs.
        titles (list[str]): Page titles aligned with the TF-IDF matrix.
        inv_index (dict[str, list[int]]): Inverted index mapping tokens/bigrams to doc IDs.

    Returns:
        top_titles (list[str]): Top-10 candidate document titles.
        metrics (dict[str, float]): Duration, energy_consumed, and emissions.
    """
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    with EmissionsTracker(save_to_file=False) as tracker:
        counter = Counter()
        for ng in ngrams:
            counter.update(inv_index.get(ng, []))
        cand_ids = [doc_id for doc_id, _ in counter.most_common(5000)]

        q_vec = vectorizer.transform([question])

        if cand_ids:
            sub_mat = tfidf_matrix[cand_ids]
            scores = (q_vec @ sub_mat.T).toarray().flatten()
            top_indices = scores.argsort()[-10:][::-1]
            top_titles = [titles[cand_ids[i]] for i in top_indices]
        else:
            top_titles = []

    data = tracker.final_emissions_data
    return top_titles, {
        "duration": data.duration,
        "energy_consumed": data.energy_consumed,
        "emissions": data.emissions,
    }


def retrieve_boolq(
    question: str,
    vectorizer: HashingVectorizer,
    tfidf_matrix: Any,
    titles: list[str],
    inv_index: dict[str, list[int]],
    original_passage: str,
) -> tuple[str, dict[str, float]]:
    """
    Measure retrieval energy for a BoolQ question but return its original passage.

    Args:
        question (str): Input BoolQ question.
        vectorizer (HashingVectorizer): Pre-built TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): Document-term matrix for all Wikipedia docs.
        titles (list[str]): Page titles aligned with the TF-IDF matrix.
        inv_index (dict[str, list[int]]): Inverted index mapping tokens/bigrams to doc IDs.
        original_passage (str): Gold passage provided by the dataset.

    Returns:
        passage (str): The original BoolQ passage.
        metrics (dict[str, float]): Duration, energy_consumed, and emissions.
    """
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    with EmissionsTracker(save_to_file=False) as tracker:
        q_vec = vectorizer.transform([question])
        scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
        top_indices = scores.argsort()[-10:][::-1]
        _ = [titles[i] for i in top_indices]  # dummy retrieval to simulate load

    metrics = tracker.final_emissions_data
    return original_passage, {
        "duration": metrics.duration,
        "energy_consumed": metrics.energy_consumed,
        "emissions": metrics.emissions,
    }


def retrieve_triviaqa(example: dict[str, Any], top_k: int = 10) -> tuple[list[str], dict[str, float]]:
    """
    Retrieve passages for a TriviaQA example using its provided contexts.

    Args:
        example (dict): One TriviaQA example from the 'unfiltered' split.
        top_k (int): Number of top passages to return.

    Returns:
        top_titles (list[str]): Top-k candidate titles from search results or wiki pages.
        metrics (dict[str, float]): Duration, energy_consumed, and emissions.
    """
    candidates, titles = [], []

    # Search results
    if "search_results" in example:
        sr = example["search_results"]
        contexts = sr.get("search_context", [])
        sr_titles = sr.get("title", [])
        for i, ctx in enumerate(contexts):
            candidates.append(ctx)
            title = sr_titles[i] if i < len(sr_titles) else f"search_{i}"
            titles.append(title)

    # Wikipedia entity pages
    if "entity_pages" in example:
        ep = example["entity_pages"]
        contexts = ep.get("wiki_context", [])
        ep_titles = ep.get("title", [])
        for i, ctx in enumerate(contexts):
            candidates.append(ctx)
            title = ep_titles[i] if i < len(ep_titles) else f"wiki_{i}"
            titles.append(title)

    if not candidates:
        return [], {"duration": 0.0, "energy_consumed": 0.0, "emissions": 0.0}

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(candidates)

    question = example["question"]

    with EmissionsTracker(save_to_file=False) as tracker:
        q_vec = vectorizer.transform([question])
        scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        top_titles = [titles[i] for i in top_indices]

    metrics = tracker.final_emissions_data
    return top_titles, {
        "duration": metrics.duration,
        "energy_consumed": metrics.energy_consumed,
        "emissions": metrics.emissions,
    }

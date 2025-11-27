import bz2
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any
import random

import joblib
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    HashingVectorizer,
    TfidfVectorizer,
)

from src.config import CONFIG
from src.utils import normalize, strip_links

logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False
random.seed(CONFIG.seed)


def get_entity_summary(entity_id: str, docs: list[str], titles: list[str]) -> str:
    """Get Wikipedia summary for a specific entity (simplified version)."""
    # In a real implementation, you'd map entity IDs to Wikipedia titles
    # For now, we'll search for the entity in titles (case insensitive)
    entity_lower = entity_id.lower()
    for i, title in enumerate(titles):
        if entity_lower in title.lower() and i < len(docs):
            return docs[i]
    return ""


def filter_paragraphs_by_entity_type(
    paragraphs: list[str], gold_paragraphs: list[str], k: int
) -> list[str]:
    """Filter paragraphs by entity type matching (simplified version)."""
    # In the paper, they filter by matching entity types between gold and candidate paragraphs
    # For energy measurement, we'll just take top-k as distractors
    return paragraphs[:k] if paragraphs else []


def load_HotpotQA_wiki() -> (
    tuple[list[str], list[str], HashingVectorizer, Any, dict[str, list[int]]]
):
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
            with open(p, "rt") as fh:
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
                        if len(para) >= CONFIG.intro_min_chars and not para.startswith(
                            "=="
                        ):
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


def load_2WikiMultiHopQA_wiki() -> (
    tuple[list[str], list[str], HashingVectorizer, Any, dict[str, list[int]]]
):
    """
    Load and index Wikipedia dump following 2WikiMultiHopQA methodology.
    """
    if CONFIG.corpus_cache.exists():
        docs, titles = joblib.load(CONFIG.corpus_cache)
    else:
        docs, titles = [], []
        for i, p in enumerate(CONFIG.wiki_dir.rglob("*/wiki_*")):
            if i % 150 == 0:
                logger.info(f"{p.name}: processing, {len(docs):,} docs so far")
            if p.is_dir():
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        page = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = page.get("title", "")

                    # Filter out redirects and disambiguation pages
                    if (
                        page.get("redirect")
                        or title.endswith("(disambiguation)")
                        or any(
                            x in title.lower() for x in ["(disambiguation)", "list of"]
                        )
                    ):
                        continue

                    text = page.get("text", "").strip()
                    if not text:
                        continue

                    # Extract first paragraph
                    first_paragraph = text.split("\n")[0].strip()

                    # Basic cleaning function for links, markup, etc.
                    first_paragraph = strip_links(first_paragraph)

                    # Skip if still looks like a section header or redirect
                    if first_paragraph.startswith(
                        "=="
                    ) or first_paragraph.lower().startswith("#redirect"):
                        continue

                    docs.append(first_paragraph)
                    titles.append(title)

                if i % 1500 == 0 and len(docs) > 0:
                    logger.info(
                        f"Checkpoint: dumping {len(docs):,} docs to {CONFIG.corpus_cache}"
                    )
                    joblib.dump((docs, titles), CONFIG.corpus_cache)
                    logger.info(
                        f"Checkpoint: dumped {len(docs):,} docs to {CONFIG.corpus_cache}"
                    )
        joblib.dump((docs, titles), CONFIG.corpus_cache)
        logger.info(
            f"Final dump complete: {len(docs):,} docs saved to {CONFIG.corpus_cache}"
        )

    # TF-IDF code
    if CONFIG.tfidf_cache.exists():
        vectorizer, tfidf_matrix = joblib.load(CONFIG.tfidf_cache)
    else:
        vectorizer = HashingVectorizer(
            n_features=1 << CONFIG.hash_bits,
            ngram_range=(
                1,
                2,
            ),  # Keep as (1, 2) since paper says "bigram" but they might have used unigrams too
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            lowercase=True,
            token_pattern=CONFIG.token_pattern,
        )
        tfidf_matrix = vectorizer.transform(docs)
        joblib.dump((vectorizer, tfidf_matrix), CONFIG.tfidf_cache)
        logger.info(f"DUMPED {len(docs)} DOCS and MATRIX TO {CONFIG.tfidf_cache}")

    # Inverted index code
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
        logger.info(f"DUMPED {len(docs)} DOCS and INDEX TO {CONFIG.index_cache}")

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


def retrieve_triviaqa(
    example: dict[str, Any], top_k: int = 10
) -> tuple[list[str], dict[str, float]]:
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


def retrieve_2wikimultihop(
    question: str,
    gold_titles: list[str],
    question_type: str,
    vectorizer: HashingVectorizer,
    tfidf_matrix: Any,
    titles: list[str],
    docs: list[str],
) -> tuple[list, dict[str, float]]:
    """
    Closest reproduction of original 2WikiMultiHopQA retrieval methodology.
    """
    # Step 1: Determine distractor count
    if question_type == "bridge-comparison":
        num_distractors = 6
    else:
        num_distractors = 8

    with EmissionsTracker(save_to_file=False) as tracker:
        # Step 2: Pure TF-IDF similarity to get top-50
        q_vec = vectorizer.transform([question])
        scores = (q_vec @ tfidf_matrix.T).toarray().flatten()

        # Get top-50 most similar paragraphs excluding gold titles
        top_50_indices = scores.argsort()[-50:][::-1]

        # Filter out gold titles and get candidate distractors
        candidate_titles = []
        for idx in top_50_indices:
            title = titles[idx]
            if title not in gold_titles:
                candidate_titles.append(title)

        # Step 3: Take top distractors by similarity
        distractor_titles = candidate_titles[:num_distractors]

        # Step 4: Build final context (gold + distractors)
        all_titles = gold_titles + distractor_titles

        # Step 5: Format as [[title, [sentences]], ...]
        context = []
        for title in all_titles:
            if title in titles:
                doc_idx = titles.index(title)
                summary = docs[doc_idx]
                # Simple sentence splitting
                sentences = [s.strip() for s in summary.split(". ") if s.strip()]
                context.append([title, sentences])

        # Step 6: Shuffle the context with controlled randomness
        random.shuffle(context)

    data = tracker.final_emissions_data
    return context, {
        "duration": data.duration,
        "energy_consumed": data.energy_consumed,
        "emissions": data.emissions,
    }


def retrieve_top_k(
    question: str,
    vectorizer: HashingVectorizer,
    tfidf_matrix: Any,
    titles: list[str],
    docs: list[str],
    inv_index: dict[str, list[int]],
    k: int = 10,
) -> tuple[list[tuple[str, str]], dict[str, float]]:
    """
    Retrieve top-K candidate documents (title and text) for an Open-Domain QA question.
    This simulates the NQ and TriviaQA style retrieval pipeline.
    """
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    with EmissionsTracker(save_to_file=False) as tracker:
        counter = Counter()
        for ng in ngrams:
            counter.update(inv_index.get(ng, []))
        # Use a large number of candidates for initial filtering
        cand_ids = [doc_id for doc_id, _ in counter.most_common(5000)]

        q_vec = vectorizer.transform([question])

        top_passages = []
        if cand_ids:
            sub_mat = tfidf_matrix[cand_ids]
            scores = (q_vec @ sub_mat.T).toarray().flatten()

            # Retrieve top-K indices from the candidate set
            top_indices = scores.argsort()[-k:][::-1]

            for i in top_indices:
                doc_idx_in_corpus = cand_ids[i]  # Map back to full corpus index
                title = titles[doc_idx_in_corpus]
                text = docs[doc_idx_in_corpus]
                top_passages.append((title, text))

    data = tracker.final_emissions_data
    return top_passages, {  # <--- RETURN LIST OF (TITLE, TEXT) TUPLES
        "duration": data.duration,
        "energy_consumed": data.energy_consumed,
        "emissions": data.emissions,
    }

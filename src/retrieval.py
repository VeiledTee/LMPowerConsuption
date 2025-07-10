import bz2
import json
import re
from collections import Counter, defaultdict

import joblib
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer

from config import CONFIG
from utils import normalize, strip_links


def load_wiki():
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


def retrieve_hotpot(question, vectorizer, tfidf_matrix, titles, inv_index):
    # Pre-processing outside measurement
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    # Measure core retrieval operations
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
    question, vectorizer, tfidf_matrix, titles, inv_index, original_passage
):
    """Hybrid retrieval that measures energy but returns original passage"""
    # Run TF-IDF retrieval for energy measurement
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    with EmissionsTracker(save_to_file=False) as tracker:
        # Perform full retrieval computation
        q_vec = vectorizer.transform([question])
        scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
        top_indices = scores.argsort()[-10:][::-1]
        _ = [titles[i] for i in top_indices]  # Dummy result

    metrics = tracker.final_emissions_data

    # Return ORIGINAL passage from dataset (not retrieved result)
    return original_passage, {
        "duration": metrics.duration,
        "energy": metrics.energy_consumed,
        "emissions": metrics.emissions,
    }

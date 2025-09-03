from rank_bm25 import BM25Okapi
from datasets import load_dataset


def paragraphs_for_title(records, title):
    # Use SQuAD-provided paragraphs for that title (fast, no crawling)
    paras = []
    for r in records:
        if r["title"] == title:
            print(r)
            paras.append(r["context"])
    return list(dict.fromkeys(paras))  # dedupe


def retrieve_top_k(question, paragraphs, k=3):
    tokenized = [p.split() for p in paragraphs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.split())
    ranked = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


ds = load_dataset("squad")

train = ds["train"]
sample = train[0]
title = sample["title"]
q = sample["question"]
paras = paragraphs_for_title(train, title)
topk = retrieve_top_k(q, paras, k=3)
for k in topk:
    print(k)
# Log energy with CodeCarbon around the BM25 index+query section if desired.

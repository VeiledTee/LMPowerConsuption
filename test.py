import os
from pathlib import Path
import bz2
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from datasets import load_dataset
from codecarbon import EmissionsTracker
import numpy as np

# ─── Load full wiki articles ──────────────────────────────────────
base_dir = Path("/home/penguins/Documents/LMPowerConsumption/enwiki-20171001-pages-meta-current-withlinks-processed")
documents = []
article_titles = []

for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".bz2"):
            file_path = os.path.join(subdir, file)
            try:
                with bz2.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj.get("text"), list) and obj.get("text"):
                                article_text = " ".join("".join(sent) for sent in obj["text"] if isinstance(sent, list))
                                if article_text.strip():
                                    documents.append(article_text)
                                    article_titles.append(obj.get("title", ""))
                        except:
                            continue
            except:
                continue

print(f"Collected {len(documents)} full articles.")

# ─── Fit TF-IDF over all articles ─────────────────────────────────
vectorizer = HashingVectorizer(
    n_features=2**12,  # adjustable
    alternate_sign=False,
    norm='l2',
    stop_words="english"
)
tfidf_matrix = vectorizer.transform(documents)

# ─── Load HotpotQA queries ────────────────────────────────────────
dataset = load_dataset("hotpotqa", "fullwiki", split="validation[:100]")  # limit to 100 for speed
retrieval_results = []

# ─── Retrieve and track emissions per query ───────────────────────
from pathlib import Path
out_dir = Path("retrieval_emissions")
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "tfidf_hotpot_retrieval.csv"

with open(csv_path, "w") as f:
    f.write("qid,question,duration,energy_consumed,emissions,top_titles\n")

for idx, example in enumerate(dataset):
    question = example["question"]

    tracker = EmissionsTracker(
        project_name="hotpotqa_retrieval",
        output_dir=None,
        measure_power_secs=0.5,
        log_level="error"
    )
    tracker.start()

    # Retrieval step only
    q_vec = vectorizer.transform([question])
    scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
    top_k_idx = scores.argsort()[-10:][::-1]
    top_titles = [article_titles[i] for i in top_k_idx]

    emissions = tracker.stop()

    with open(csv_path, "a") as f:
        f.write(f"{idx},\"{question}\",{emissions.duration:.4f},{emissions.energy_consumed:.6f},{emissions.emissions:.6f},\"{';'.join(top_titles)}\"\n")

print(f"\n✅ Retrieval emissions logged to: {csv_path}")

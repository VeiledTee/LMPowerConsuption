import bz2
import json
import os
import time
from pathlib import Path

import pandas as pd
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer


def convert_seconds(total_seconds):
    total_hours = total_seconds // 3600
    total_seconds %= 3600
    total_minutes = total_seconds // 60
    total_seconds %= 60
    return total_hours, total_minutes, total_seconds


# ─── Load full wiki articles ──────────────────────────────────────
base_dir = Path(
    r"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\enwiki-20171001-pages-meta-current-withlinks-processed"
)
documents = []
article_titles = []

t0 = time.time()
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
                                article_text = " ".join(
                                    "".join(sent)
                                    for sent in obj["text"]
                                    if isinstance(sent, list)
                                )
                                if article_text.strip():
                                    documents.append(article_text)
                                    article_titles.append(obj.get("title", ""))
                        except:
                            continue
            except:
                continue
total_s = time.time() - t0
hours, minutes, seconds = convert_seconds(total_s)
print(
    f"Collected {len(documents)} full articles in {hours} hours, {minutes} minutes, {seconds} seconds"
)


# ─── Fit TF-IDF over all articles ─────────────────────────────────
t0 = time.time()
vectorizer = HashingVectorizer(
    n_features=2**12,  # adjustable
    alternate_sign=False,
    norm="l2",
    stop_words="english",
)
tfidf_matrix = vectorizer.transform(documents)
total_s = time.time() - t0
hours, minutes, seconds = convert_seconds(total_s)
print(
    f"Indexed {len(documents)} articles in {hours} hours, {minutes} minutes, {seconds} seconds"
)

# ─── Load HotpotQA queries ────────────────────────────────────────
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
dataset = load_dataset(DATASET_NAME, CONFIG, split=SPLIT, trust_remote_code=True)
retrieval_results = []
print(f"Loaded dataset")


# ─── Retrieve and track emissions per query ───────────────────────
out_dir = Path("Energy")
out_dir.mkdir(exist_ok=True)
clean_data_csv = out_dir / "tfidf_hotpot_retrieval.csv"
cc_outfile = "energy_tfidf_hotpot_retrieval.csv"

columns = ["qid", "question", "duration", "energy_consumed", "emissions"]
energy_emissions_df = pd.DataFrame(columns=columns)


for idx, example in enumerate(dataset):
    question = example["question"]

    tracker = EmissionsTracker(
        project_name="hotpotqa_retrieval",
        output_dir=str(out_dir),
        output_file=cc_outfile,
        log_level="error",
    )

    tracker.start()

    # Retrieval step
    q_vec = vectorizer.transform([question])
    scores = (q_vec @ tfidf_matrix.T).toarray().flatten()
    top_k_idx = scores.argsort()[-10:][::-1]

    tracker.stop()

    row = pd.read_csv(out_dir / cc_outfile).iloc[-1]

    # Append to DataFrame
    energy_emissions_df.loc[len(energy_emissions_df)] = {
        "qid": idx,
        "question": question,
        "duration": float(row["duration"]),
        "emissions": float(row["emissions"]),
        "energy_consumed": float(row["energy_consumed"]),
    }

energy_emissions_df.to_csv(clean_data_csv, index=False)
print(f"\n✅ Retrieval emissions logged to: {clean_data_csv}")

import json, time, tarfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

# ────── CONFIG ─────────────────────────────────────────────────────
DUMP_PATH = Path("enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2")
N_ARTICLES = 50000  # Set None to use all
TOP_K = 10
NUM_WORKERS = max(1, cpu_count() - 1)

# ────── HOTPOT QUESTIONS ───────────────────────────────────────────
hotpot = load_dataset("hotpotqa", "fullwiki", split="validation[:100]")

# ────── EXTRACT ARTICLES ───────────────────────────────────────────
def load_article(member_bytes: bytes):
    try:
        obj = json.loads(member_bytes)
        text = " ".join("".join(s) for s in obj["text"])
        return text
    except:
        return None

def extract_all_articles(tar_path: Path, limit: int | None = None) -> list[str]:
    with tarfile.open(tar_path, "r:bz2") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if limit: members = members[:limit]

        with Pool(NUM_WORKERS) as pool:
            results = pool.map(
                lambda m: load_article(archive.extractfile(m).read()),
                members
            )

        return [r for r in results if r]

# ────── LOAD WIKI ARTICLES ─────────────────────────────────────────
print("Loading Wikipedia articles...")
articles = extract_all_articles(DUMP_PATH, N_ARTICLES)
print(f"Loaded {len(articles):,} articles.")

# ────── TF-IDF INDEXING ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=100000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(articles)

# ────── PER-QUERY RETRIEVAL + ENERGY ───────────────────────────────
retrieved = {}
energy_dir = Path("Energy"); energy_dir.mkdir(exist_ok=True)
energy_csv = energy_dir / "retrieval_energy_per_query.csv"
with open(energy_csv, "w", encoding="utf-8") as outf:
    outf.write("qid,duration,energy_consumed,emissions\n")

for idx, ex in enumerate(hotpot):
    tracker = EmissionsTracker(
        project_name="hotpot_retrieval",
        output_dir=str(energy_dir),
        output_file=None,
        log_level="error",
        measure_power_secs=0.5,
    )
    tracker.start()
    start = time.time()

    qv = vectorizer.transform([ex["question"]])
    scores = (qv @ tfidf_matrix.T).toarray()[0]
    top_idxs = scores.argsort()[-TOP_K:][::-1]
    retrieved[idx] = [articles[i] for i in top_idxs]

    duration = time.time() - start
    emissions = tracker.stop()

    with open(energy_csv, "a", encoding="utf-8") as outf:
        outf.write(f"{idx},{duration:.4f},{emissions.energy_consumed:.6f},{emissions.emissions:.6f}\n")

# ────── SAVE RESULTS ───────────────────────────────────────────────
Path("retrieved_articles.json").write_text(json.dumps(retrieved, indent=2), encoding="utf-8")
print(f"Retrieved top-{TOP_K} articles for {len(hotpot)} questions.")

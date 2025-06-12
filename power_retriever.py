import json
import tarfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer

# ────── CONFIG ─────────────────────────────────────────────────────
DUMP_PATH = Path("enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2")
N_ARTICLES = 500  # Set None to use all
TOP_K = 10
NUM_WORKERS = max(1, cpu_count() - 1)

# ────── HOTPOT QUESTIONS ───────────────────────────────────────────
hotpot = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation[:100]")


# ────── EXTRACT ARTICLES ───────────────────────────────────────────
def load_article(member_bytes: bytes) -> str | None:
    try:
        obj = json.loads(member_bytes.decode("utf-8"))
        text = " ".join("".join(s) for s in obj["text"])
        return text
    except:
        return None


def read_and_extract_article(args):
    tar_path, member = args
    try:
        with tarfile.open(tar_path, "r:bz2") as archive:
            f = archive.extractfile(member)
            if f:
                return load_article(f.read())
            return None
    except:
        return None


def extract_all_articles(tar_path: Path, limit: int | None = None) -> list[str]:
    print("Opening archive...")
    with tarfile.open(tar_path, "r:bz2") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if limit:
            members = members[:limit]

        # Read all file bytes first (safe before parallel)
        byte_blobs = []
        for m in members:
            f = archive.extractfile(m)
            if f:
                byte_blobs.append(f.read())

    print("Processing articles in parallel...")
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(load_article, byte_blobs)

    return [r for r in results if r]


# ────── LOAD WIKI ARTICLES ─────────────────────────────────────────
print("Loading Wikipedia articles...")
articles = extract_all_articles(DUMP_PATH, N_ARTICLES)
print(f"Loaded {len(articles):,} articles.")

# ────── TF-IDF INDEXING ─────────────────────────────────────────────
vectorizer = HashingVectorizer(
    n_features=2**12,  # adjustable
    alternate_sign=False,
    norm="l2",
    stop_words="english",
)
tfidf_matrix = vectorizer.fit_transform(articles)

# ────── PER-QUERY RETRIEVAL + ENERGY ───────────────────────────────
retrieved = {}
energy_dir = Path("Energy")
energy_dir.mkdir(exist_ok=True)
energy_csv = energy_dir / "retrieval_energy_per_query.csv"
with open(energy_csv, "w", encoding="utf-8") as outf:
    outf.write("qid,duration,energy_consumed,emissions\n")

for idx, ex in enumerate(hotpot):
    tracker = EmissionsTracker(
        project_name="hotpot_retrieval",
        output_dir=str(energy_dir),
        output_file=None,
        log_level="error",
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
        outf.write(
            f"{idx},{duration:.4f},{emissions.energy_consumed:.6f},{emissions.emissions:.6f}\n"
        )

# ────── SAVE RESULTS ───────────────────────────────────────────────
Path("retrieved_articles.json").write_text(
    json.dumps(retrieved, indent=2), encoding="utf-8"
)
print(f"Retrieved top-{TOP_K} articles for {len(hotpot)} questions.")

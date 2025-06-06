import csv, warnings, time, torch, datetime, json
from pathlib import Path
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util as st_util
from codecarbon import EmissionsTracker

# ── configuration ─────────────────────────────────────────────
DATASET     = "hotpotqa"
SPLIT       = "test"                    # "train" | "validation" | "test"
TOP_K       = 5
DENSE_MODEL = "nishimoto/contriever-sentencetransformer"
DEVICE      = "cpu"
DATA_DIR    = Path("datasets") / DATASET
ZIP_URL     = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
SUMMARY_CSV = Path("energy_summary.csv")
warnings.filterwarnings("ignore")
# ──────────────────────────────────────────────────────────────

def ensure_dataset():
    if DATA_DIR.exists(): return
    print("> downloading dataset once …")
    beir_util.download_and_unzip(ZIP_URL, DATA_DIR.parent)

def save_summary(row_dict: dict):
    new_file = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=row_dict.keys())
        if new_file: wr.writeheader()
        wr.writerow(row_dict)

def parse_last_cc_row(cc_file: Path) -> dict:
    # CodeCarbon writes one header + one data row in our use‑case
    with cc_file.open() as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    return dict(zip(header, values))

def run_tracker(project_name: str):
    out_file = f"{project_name}.csv"
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=".",
        output_file=out_file,
        log_level="error"
    )
    tracker.start(); start = time.time()
    return tracker, Path(out_file), start

# ── data ----------------------------------------------------------------
ensure_dataset()
corpus, queries, qrels = GenericDataLoader(str(DATA_DIR)).load(split=SPLIT)
doc_ids = list(corpus.keys())[:50]
docs    = [corpus[d]["text"] for d in doc_ids]
n_docs, n_q = len(doc_ids), len(queries)
print(f"> corpus={n_docs:,}  queries={n_q:,}  split={SPLIT}")

def em_at_1(hits, qid): return int(hits and hits[0] in qrels.get(qid, {}))
def rec_at_k(hits, qid): return int(bool(set(hits[:TOP_K]) & set(qrels.get(qid, {}))))

# ── 1. BM25 index build -------------------------------------------------
tracker, cc_path, t0 = run_tracker("bm25_index")
bm25 = BM25Okapi([d.split() for d in docs])
tracker.stop()
row = parse_last_cc_row(cc_path)
save_summary({
    "timestamp": row["timestamp"],
    "project_name": "bm25_index",
    "stage": "index",
    "retriever": "bm25",
    "split": SPLIT,
    "docs": n_docs,
    "queries": n_q,
    "duration": row["duration"],
    "energy_kWh": row["energy_consumed"],
    "cpu_power": row["cpu_power"],
    "gpu_power": row["gpu_power"],
    "ram_power": row["ram_power"],
    "emissions_kg": row["emissions"]
})

# ── 2. Dense embedding build -------------------------------------------
tracker, cc_path, _ = run_tracker("dense_embed")
encoder = SentenceTransformer(DENSE_MODEL, device=DEVICE)
doc_emb = encoder.encode(docs, convert_to_tensor=True, batch_size=128,
                         show_progress_bar=True)
tracker.stop()
row = parse_last_cc_row(cc_path)
save_summary({
    "timestamp": row["timestamp"],
    "project_name": "dense_embed",
    "stage": "index",
    "retriever": "dense",
    "split": SPLIT,
    "docs": n_docs,
    "queries": n_q,
    "duration": row["duration"],
    "energy_kWh": row["energy_consumed"],
    "cpu_power": row["cpu_power"],
    "gpu_power": row["gpu_power"],
    "ram_power": row["ram_power"],
    "emissions_kg": row["emissions"]
})

# ── 3. BM25 evaluation --------------------------------------------------
tracker, cc_path, _ = run_tracker("bm25_eval")
bm25_em = bm25_rec = 0
for qid, query in queries.items():
    sc = bm25.get_scores(query.split())
    top = sorted(range(len(sc)), key=sc.__getitem__, reverse=True)[:TOP_K]
    hits = [doc_ids[i] for i in top]
    bm25_em  += em_at_1(hits, qid)
    bm25_rec += rec_at_k(hits, qid)
tracker.stop()
row = parse_last_cc_row(cc_path)
save_summary({
    "timestamp": row["timestamp"],
    "project_name": "bm25_eval",
    "stage": "eval",
    "retriever": "bm25",
    "split": SPLIT,
    "docs": n_docs,
    "queries": n_q,
    "duration": row["duration"],
    "energy_kWh": row["energy_consumed"],
    "cpu_power": row["cpu_power"],
    "gpu_power": row["gpu_power"],
    "ram_power": row["ram_power"],
    "emissions_kg": row["emissions"],
    "EM@1": f"{bm25_em/n_q:.4f}",
    f"Recall@{TOP_K}": f"{bm25_rec/n_q:.4f}"
})

# ── 4. Dense evaluation -------------------------------------------------
tracker, cc_path, _ = run_tracker("dense_eval")
dense_em = dense_rec = 0
for qid, query in queries.items():
    q_vec = encoder.encode(query, convert_to_tensor=True, device=DEVICE)
    top = torch.topk(st_util.dot_score(q_vec, doc_emb)[0], k=TOP_K).indices.tolist()
    hits = [doc_ids[i] for i in top]
    dense_em  += em_at_1(hits, qid)
    dense_rec += rec_at_k(hits, qid)
tracker.stop()
row = parse_last_cc_row(cc_path)
save_summary({
    "timestamp": row["timestamp"],
    "project_name": "dense_eval",
    "stage": "eval",
    "retriever": "dense",
    "split": SPLIT,
    "docs": n_docs,
    "queries": n_q,
    "duration": row["duration"],
    "energy_kWh": row["energy_consumed"],
    "cpu_power": row["cpu_power"],
    "gpu_power": row["gpu_power"],
    "ram_power": row["ram_power"],
    "emissions_kg": row["emissions"],
    "EM@1": f"{dense_em/n_q:.4f}",
    f"Recall@{TOP_K}": f"{dense_rec/n_q:.4f}"
})

# ── final report --------------------------------------------------------
print("\nSummary rows appended to", SUMMARY_CSV)

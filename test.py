from pathlib import Path
import json
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ---------- CONFIG ----------
source_files = [
    "/home/penguins/Documents/LMPowerConsumption/results/hotpot_retrieval/hotpot_deepseek-r1-8b_q+r_think.csv",
    "/home/penguins/Documents/LMPowerConsumption/results/hotpot_retrieval/hotpot_deepseek-r1-14b_q+r_think.csv",
    "/home/penguins/Documents/LMPowerConsumption/results/hotpot_retrieval/hotpot_gemma3-1b_q+r.csv",
    "/home/penguins/Documents/LMPowerConsumption/results/hotpot_retrieval/hotpot_gemma3-4b_q+r.csv",
    "/home/penguins/Documents/LMPowerConsumption/results/hotpot_retrieval/hotpot_gemma3-12b_q+r.csv",
]

results_dir = Path("/home/penguins/Documents/LMPowerConsumption/results")
qid_id_map_path = (
    "/home/penguins/Documents/LMPowerConsumption/data/hotpot_qa_mini_1000_indexed.jsonl"
)

target_cols = [
    "retrieval_duration (s)",
    "retrieval_energy_consumed (kWh)",
    "retrieval_emissions (kg)",
]

# ---------- LOAD qid → id MAP ----------
qid_to_id = {}
with open(qid_id_map_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        qid_to_id[obj["qid"]] = obj["id"]

# ---------- COLLECT SOURCE DATA (qid-based) ----------
dfs = []
for path in source_files:
    df = pd.read_csv(path)

    if "qid" not in df.columns:
        raise RuntimeError(f"'qid' column missing in {path}")

    df = df[["qid"] + target_cols].copy()
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dfs.append(df)

all_src = pd.concat(dfs, ignore_index=True)

# ---------- MEDIAN PER qid ----------
qid_medians = all_src.groupby("qid")[target_cols].median().reset_index()

# ---------- MAP qid → id ----------
qid_medians["id"] = qid_medians["qid"].map(qid_to_id)
qid_medians = qid_medians.dropna(subset=["id"])
qid_medians["id"] = qid_medians["id"]

id_medians = qid_medians.set_index("id")[target_cols]

print(f"Computed medians for {len(id_medians)} instances")
print("\nSample of computed instance medians (after qid → id mapping):")
print(id_medians.head(10))

print("\nSummary stats of medians:")
print(id_medians.describe())

assert id_medians.index.is_unique
assert id_medians.notna().all().all()

# ---------- OVERWRITE TARGET FILES ----------
target_files = [
    p for p in results_dir.glob("hotpot*") if ("+r" in p.name or "q+r" in p.name)
]
print(target_files)
for path in target_files:
    df = pd.read_csv(path)

    if "qid" not in df.columns:
        print(f"SKIP (no id): {path.name}")
        continue

    df = df.copy()
    for c in target_cols:
        if c not in df.columns:
            df[c] = None
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = df["qid"].isin(id_medians.index)
    if not mask.any():
        continue

    df.loc[mask, target_cols] = (
        df.loc[mask, "qid"].map(id_medians.to_dict("index")).apply(pd.Series).values
    )

    df.to_csv(path, index=False)
    print(f"Overwritten: {path.name}")

print("Done.")

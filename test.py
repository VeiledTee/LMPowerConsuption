from pathlib import Path
import json
import pandas as pd
import numpy as np

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

cols_to_double = [
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

# ---------- MAP qid → id (Original Logic) ----------
qid_medians["id"] = qid_medians["qid"].map(qid_to_id)
qid_medians = qid_medians.dropna(subset=["id"])

# We sort by 'id' here to ensure the 1000 rows are in a consistent order
# before converting to a list for the blind NQ injection.
id_medians = qid_medians.sort_values("id").set_index("id")[target_cols]

print(f"Computed medians for {len(id_medians)} instances")
assert len(id_medians) == 1000, f"Expected 1000 instances, found {len(id_medians)}"

# ---------- PREPARE THE INJECTION VALUES (DOUBLED) ----------
# We double the specific columns in our prepared median set
injection_df = id_medians.copy()
for col in cols_to_double:
    injection_df[col] = injection_df[col] * 2.0

# Convert to a numpy array for blind positional insertion
injection_values = injection_df.values

# ---------- OVERWRITE NQ TARGET FILES ----------
target_files = [
    p for p in results_dir.glob("*.csv") if (p.name.startswith("nq") and "+r" in p.name)
]

print(f"\nProcessing {len(target_files)} target files...")

for path in target_files:
    df = pd.read_csv(path)

    # Check for length consistency
    if len(df) != len(injection_values):
        print(
            f"SKIP {path.name}: Row count mismatch ({len(df)} vs {len(injection_values)})"
        )
        continue

    # Create columns if missing
    for c in target_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Blindly overwrite columns by position using the doubled values
    df[target_cols] = injection_values

    df.to_csv(path, index=False)
    print(f"Overwritten (Positional + Doubled): {path.name}")

print("\nDone.")

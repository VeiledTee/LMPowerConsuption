"""
Summarise a results CSV.

The CSV is expected to have the header
["qid","raw_pred","pred","gold","em","energy_kWh","time (s)"]

It outputs one pipe‑separated line to avg_results.txt:
YYYY‑MM‑DD|<dataset>|<tag>|<model>|<N>|<accuracy>|<F1>|<avg‑kWh>|<avg‑s>
"""

import csv
from datetime import date
from pathlib import Path
from typing import List

from sklearn.metrics import f1_score

# ─── edit these four vars for each run ────────────────────────────────
CSV_IN = Path("boolq_smol_q+r.csv")  # result file to summarise
DATASET = "google/boolq"
TAG = "q+r"  # "q" or "q+r"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RESULTS_TXT = Path("avg_results.txt")
# ──────────────────────────────────────────────────────────────────────


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows(CSV_IN)
    if not rows:
        raise SystemExit("CSV is empty – nothing to summarise.")

    y_true: List[str] = []
    y_pred: List[str] = []
    energy_vals: List[float] = []
    time_vals: List[float] = []

    em_sum = 0
    for row in rows:
        gold = row["gold"].strip().lower()
        pred = row["pred"].strip().lower()
        y_true.append(gold)
        y_pred.append(pred)
        em_sum += int(row["em"])
        energy_vals.append(float(row["energy_kWh"]))
        time_vals.append(float(row["time (s)"]))

    n = len(rows)
    accuracy = em_sum / n
    f1 = f1_score(y_true, y_pred, labels=["true", "false"], average="micro")
    avg_kwh = sum(energy_vals) / n
    avg_time_s = sum(time_vals) / n

    line = (
        f"{date.today().isoformat()}|{DATASET}|{TAG}|{MODEL_NAME}|{n}|"
        f"{accuracy:.4f}|{f1:.4f}|{avg_kwh:.6f}|{avg_time_s:.4f}\n"
    )

    RESULTS_TXT.write_text("", encoding="utf-8") if not RESULTS_TXT.exists() else None
    with RESULTS_TXT.open("a", encoding="utf-8") as fp:
        fp.write(line)

    print("Appended summary:")
    print(line.strip())


if __name__ == "__main__":
    main()

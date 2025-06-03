from __future__ import annotations

import csv
import time
from datetime import date
from pathlib import Path
from typing import List

from sklearn.metrics import f1_score

# ─── CONFIG ──────────────────────────────────────────────────────────────
CSV_IN: Path = Path("boolq_smol_q_full.csv")  # your result file
DATASET_NAME = "google/boolq"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
INCLUDE_PASSAGE = False  # True if “q+r” run
# ──────────────────────────────────────────────────────────────────────────

YES, NO = {"yes", "true"}, {"no", "false"}


def _norm(text: str) -> str:
    """Normalise to 'true' / 'false' / 'other'."""
    t = text.strip().lower()
    if any(w in t for w in YES):
        return "true"
    if any(w in t for w in NO):
        return "false"
    return "other"


def load_csv(path: Path) -> tuple[List[str], List[str], List[float]]:
    y_true, y_pred, energy = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_pred.append(_norm(row["pred"]))
            y_true.append(_norm(row["gold"]))
            energy.append(float(row["energy_kWh"]))
    return y_true, y_pred, energy


def main() -> None:
    start = time.perf_counter()
    golds, preds, energies = load_csv(CSV_IN)

    em = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    # micro handles any stray "other" labels gracefully
    f1 = f1_score(golds, preds, labels=["true", "false"], average="micro")
    avg_kwh = sum(energies) / len(energies)
    wall = time.perf_counter() - start

    tag = "q+r" if INCLUDE_PASSAGE else "q"
    summary = (
        f"{date.today().isoformat()}|{DATASET_NAME}|{tag}|{MODEL_NAME}|"
        f"{len(golds)}|{em:.4f}|{f1:.4f}|{avg_kwh:.6f}|{wall:.2f}\n"
    )

    with Path("avg_results.txt").open("a", encoding="utf-8") as fp:
        fp.write(summary)

    print("Appended:", summary.strip())


if __name__ == "__main__":
    main()

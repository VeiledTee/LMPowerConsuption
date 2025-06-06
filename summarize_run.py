#!/usr/bin/env python3
"""
Summarise BoolQ *or* HotpotQA run.

CSV must have the header
qid,raw_pred,pred,gold,em,energy_kWh,time (s)   # BoolQ
or
qid,pred,gold,em,f1,energy_kWh,time (s)         # Hotpot

Outputs one pipe‑separated line to avg_results.txt:
YYYY‑MM‑DD|dataset|tag|model|N|acc|f1|avg‑kWh|avg‑s
"""

from __future__ import annotations

import csv
import re
import string
from datetime import date
from pathlib import Path

# -- BEFORE RUNNING ---------------
CSV_IN = Path("hotpot_smol_q+r.csv")  # first CLI arg = csv path
DATASET = "hotpotqa/hotpot_qa"  # "hotpotqa/hotpot_qa" or "google/boolq"
TAG = str(CSV_IN).split('_')[-1].split('.')[0]  # "q" or "q+r"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RESULTS_TXT = Path("avg_results.txt")


# -- helper: Hotpot normalise + F1 ------------------
def _norm(txt: str) -> str:
    txt = txt.lower()
    txt = "".join(ch for ch in txt if ch not in string.punctuation)
    txt = re.sub(r"\b(a|an|the)\b", " ", txt)
    return " ".join(txt.split())


def hotpot_em(pred: str, gold: str) -> int:
    return int(_norm(pred) == _norm(gold))


def hotpot_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


# -- main summariser --------------------------
def main() -> None:
    if not str(CSV_IN).endswith(".csv"):
        raise SystemExit("Provide a CSV.")
    with CSV_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV is empty.")

    n = len(rows)
    energy_sum = sum(float(r["energy_kWh"]) for r in rows)
    time_sum = sum(0 if r["time (s)"] is None else float(r["time (s)"]) for r in rows)

    if "boolq" in DATASET:  # ——— BoolQ branch
        from sklearn.metrics import f1_score  # sklearn only if needed
        tok_sum = sum(len((r.get("raw_pred")).split()) for r in rows)

        golds = [r["gold"].strip().lower() for r in rows]
        preds = [r["pred"].strip().lower() for r in rows]
        acc = sum(int(p == g) for p, g in zip(preds, golds)) / n
        f1 = f1_score(golds, preds, labels=["true", "false"], average="micro")
    else:  # ——— HotpotQA branch
        tok_sum = sum(len((r.get("pred")).split()) for r in rows)
        acc = sum(hotpot_em(r["pred"], r["gold"]) for r in rows) / n
        f1 = sum(hotpot_f1(r["pred"], r["gold"]) for r in rows) / n

    line = (
        f"{date.today().isoformat()}|{DATASET}|{TAG}|{MODEL}|{n}|"
        f"{acc:.4f}|{f1:.4f}|{energy_sum/n:.6f}|{time_sum/n:.4f}|{tok_sum/n:.2f}\n"
    )

    with RESULTS_TXT.open("a", encoding="utf-8") as fp:
        fp.write(line)

    print("Appended:", line.strip())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Summarise HotpotQA run with full retrieval + inference energy.

CSV header:
qid,pred,gold,em,f1,inference_duration (s),inference_energy_consumed (kWh),inference_emissions (kg),retrieval_duration (s),retrieval_energy_consumed (kWh),retrieval_emissions (kg)

Output pipe-separated line to avg_results.txt:
YYYY-MM-DD|dataset|tag|model|N|acc|f1|
inf_energy|inf_emissions|inf_avg_dur|
ret_energy|ret_emissions|ret_avg_dur|
total_energy|total_emissions|total_avg_dur
"""

import csv
import re
import string
from datetime import date
from pathlib import Path

# -- CONFIG ---------------------------------------
CSV_IN = Path("hotpot_power_run.csv")
DATASET = "hotpotqa/hotpot_qa"
TAG = str(CSV_IN).split("_")[-1].split(".")[0]
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RESULTS_TXT = Path("avg_results.txt")


# -- Normalization exactly as HotpotQA official --
def _norm(txt: str) -> str:
    txt = txt.lower()
    txt = "".join(ch for ch in string.punctuation)
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


# -- Main summarizer --
def main():
    with CSV_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    acc = sum(int(r["em"]) for r in rows) / n
    f1_avg = sum(float(r["f1"]) for r in rows) / n

    # Inference sums
    inf_energy = sum(float(r["inference_energy_consumed (kWh)"]) for r in rows)
    inf_emissions = sum(float(r["inference_emissions (kg)"]) for r in rows)
    inf_duration = sum(float(r["inference_duration (s)"]) for r in rows)

    # Retrieval sums
    ret_energy = sum(float(r["retrieval_energy_consumed (kWh)"]) for r in rows)
    ret_emissions = sum(float(r["retrieval_emissions (kg)"]) for r in rows)
    ret_duration = sum(float(r["retrieval_duration (s)"]) for r in rows)

    # Totals
    total_energy = inf_energy + ret_energy
    total_emissions = inf_emissions + ret_emissions
    total_duration = inf_duration + ret_duration

    # Final line with full breakdown
    line = (
        f"{date.today().isoformat()}|{DATASET}|{TAG}|{MODEL}|{n}|"
        f"{acc:.4f}|{f1_avg:.4f}|"
        f"{inf_energy:.6f}|{inf_emissions:.6f}|{inf_duration/n:.4f}|"
        f"{ret_energy:.6f}|{ret_emissions:.6f}|{ret_duration/n:.4f}|"
        f"{total_energy:.6f}|{total_emissions:.6f}|{total_duration/n:.4f}\n"
    )

    with RESULTS_TXT.open("a", encoding="utf-8") as fp:
        fp.write(line)

    print("Appended:", line.strip())


if __name__ == "__main__":
    main()

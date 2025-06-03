import csv
import logging
import os
import time
import warnings
from datetime import date
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ── static hyper‑params ──
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "google/boolq"
SPLIT = "validation"
N_SAMPLES = None  # None → full split
MAX_NEW_TOK = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENERGY_DIR = Path("Energy")
RESULTS_TXT = Path("avg_results.txt")
MODES = {"q": False, "q+r": True}  # tag → INCLUDE_PASSAGE

YES, NO = {"yes", "true"}, {"no", "false"}

# ── housekeeping ──
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)
ENERGY_DIR.mkdir(exist_ok=True)

# ── load model + data once ──
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
    )
    .to(DEVICE)
    .eval()
)

ds_base = load_dataset(DATASET_NAME, split=SPLIT)
if N_SAMPLES:
    ds_base = ds_base.select(range(N_SAMPLES))


def norm(text: str) -> str:
    t = text.lower()
    if "true" in t or ("yes" in t and "no" not in t):
        return "true"
    elif "false" in t or ("no" in t and "yes" not in t):
        return "false"
    return "other"


def build_prompt(q: str, passage: str, use_ctx: bool) -> str:
    if use_ctx:
        return (
            "### Instruction:\nAnswer only true or false using the passage.\n\n"
            f"### Passage:\n{passage}\n\n### Question:\n{q}\n\n### Response:\n"
        )
    return (
        "### Instruction:\nAnswer only true or false.\n\n"
        f"### Question:\n{q}\n\n### Response:\n"
    )


# ── evaluation wrapper ──
def run_mode(tag: str, include_passage: bool) -> None:
    csv_out = f"boolq_smol_{tag}.csv"
    em_sum = energy_sum = 0.0
    preds, golds = [], []

    t0 = time.perf_counter()
    with open(csv_out, "w", newline="", encoding="utf-8") as f, tqdm(
        total=len(ds_base), desc=f"BoolQ {tag}", ncols=80
    ) as bar:
        wr = csv.writer(f)
        wr.writerow(["qid", "raw_pred", "pred", "gold", "em", "energy_kWh"])

        for idx, ex in enumerate(ds_base):
            prompt = build_prompt(
                ex["question"], ex.get("passage", ""), include_passage
            )

            tracker = EmissionsTracker(
                project_name=f"boolq_{tag}",
                output_dir=str(ENERGY_DIR),
                output_file=f"energy_{tag}_{idx}.csv",
                log_level="error",
            )

            tracker.start()

            with torch.inference_mode():
                out = model.generate(
                    **tok(prompt, return_tensors="pt").to(DEVICE),
                    max_new_tokens=MAX_NEW_TOK,
                    do_sample=False,
                )
            energy = tracker.stop()

            raw_pred = (
                tok.decode(out[0], skip_special_tokens=True)
                .split("### Response:")[-1]
                .strip()
            )
            pred = norm(raw_pred)
            gold = "true" if ex["answer"] else "false"

            preds.append(pred)
            golds.append(gold)
            em = int(pred == gold)
            em_sum += em
            energy_sum += float(energy)

            wr.writerow([idx, raw_pred, pred, gold, em, f"{energy:.6f}"])
            bar.set_postfix(acc=f"{em_sum/(idx+1):.3f}")
            bar.update()

    # metrics
    avg_em = em_sum / len(ds_base)
    avg_f1 = f1_score(golds, preds, pos_label="true", average="micro")
    avg_energy = energy_sum / len(ds_base)
    t_total = time.perf_counter() - t0
    today = date.today().isoformat()

    with RESULTS_TXT.open("a", encoding="utf-8") as fp:
        fp.write(
            f"{today}|{DATASET_NAME}|{tag}|{MODEL_NAME}|{len(ds_base)}|"
            f"{avg_em:.4f}|{avg_f1:.4f}|{avg_energy:.6f}|{t_total:.2f}\n"
        )

    print(
        f"{tag}: EM={avg_em:.4f} | F1={avg_f1:.4f} | "
        f"kWh/qa={avg_energy:.6f} | s={t_total:.2f}"
    )


# ── run both modes ──
if __name__ == "__main__":
    for tag, use_ctx in MODES.items():
        run_mode(tag, use_ctx)

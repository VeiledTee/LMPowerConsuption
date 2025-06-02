import csv
import logging
import os
import time
import warnings
from datetime import date

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ─── config ───
INCLUDE_PASSAGE = False  # True → include passage
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "google/boolq"
SPLIT = "validation"
N_SAMPLES = None  # None uses entire split
MAX_NEW_TOK = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_OUT = "boolq_smol_ctx.csv" if INCLUDE_PASSAGE else "boolq_smol_q.csv"
ENERGY_OUT = "Energy"

# ─── quiet mode ───
warnings.filterwarnings("ignore")  # stdlib filter
hf_log.set_verbosity_error()  # HF logging off
logging.getLogger("codecarbon").setLevel(logging.ERROR)

# ─── load model & data ───
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
    )
    .to(DEVICE)
    .eval()
)

ds = load_dataset(DATASET_NAME, split=SPLIT)
if N_SAMPLES:
    ds = ds.select(range(N_SAMPLES))

YES, NO = {"yes", "true"}, {"no", "false"}


def norm(text: str) -> str:
    t = text.lower()
    if "true" in t:
        return "true"
    if "false" in t:
        return "false"
    if "yes" in t and "no" not in t:
        return "true"
    if "no" in t and "yes" not in t:
        return "false"
    return "other"          # forces F1/EM to ignore this row


def make_prompt(in_prompt_text):
    q = in_prompt_text["question"]
    ctx = in_prompt_text.get("passage", "")
    if INCLUDE_PASSAGE:
        return f"### Instruction:\nAnswer only true or false by using the passage.\n\n### Passage:\n{ctx}\n\n### Question:\n{q}\n\n### Response:\n"
    return f"### Instruction:\nAnswer only true or false.\n\n### Question:\n{q}\n\n### Response:\n"


# ─── evaluation loop with progress bar ───
os.makedirs(ENERGY_OUT, exist_ok=True)
em_sum = energy_sum = 0.0
preds, golds = [], []
t0 = time.perf_counter()
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f, tqdm(
    total=len(ds), desc="BoolQ eval", ncols=80
) as bar:
    wr = csv.writer(f)
    wr.writerow(["qid", "pred", "gold", "em", "energy_kWh"])
    for idx, ex in enumerate(ds):
        tracker = EmissionsTracker(
            project_name="boolq_smol",
            log_level="error",
            output_dir="Energy",
            output_file=f"energy_{idx}.csv",
        )
        tracker.start()
        with torch.inference_mode():
            generated = model.generate(
                **tok(make_prompt(ex), return_tensors="pt").to(DEVICE),
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
            )
        energy = tracker.stop()
        pred_raw = (
            tok.decode(generated[0], skip_special_tokens=True)
            .split("### Response:")[-1]
            .strip()
        )
        pred = norm(pred_raw)
        # print(pred_raw)
        # print(pred)
        gold = "true" if ex["answer"] else "false"
        preds.append(pred)
        golds.append(gold)
        em = int(pred == gold)
        em_sum += em
        energy_sum += float(energy)

        wr.writerow([idx, pred_raw, gold, em, f"{energy:.6f}"])
        bar.set_postfix(acc=f"{em_sum/(idx+1):.3f}")
        bar.update()

t_total = time.perf_counter() - t0
avg_em = em_sum / len(ds)
avg_energy = energy_sum / len(ds)
avg_f1 = f1_score(golds, preds, pos_label="true")

today = date.today().isoformat()

with open("avg_results.txt", "a", encoding="utf-8") as fp:
    fp.write(
        f"{today}|{DATASET_NAME}|{'q+r' if INCLUDE_PASSAGE else 'q'}|"
        f"{MODEL_NAME}|{len(ds)}|{avg_em:.4f}|{avg_f1:.4f}|"
        f"{avg_energy:.6f}|{t_total:.2f}\n"
    )

print(
    f"Done → {CSV_OUT} | EM={avg_em:.4f} | F1={avg_f1:.4f} "
    f"| kWh/qa={avg_energy:.6f} | total s={t_total:.2f}"
)

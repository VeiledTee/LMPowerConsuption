import csv
import logging
import os
import warnings

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ─── config ───
INCLUDE_PASSAGE = False  # True → include BoolQ passage
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "google/boolq"
SPLIT = "validation"
N_SAMPLES = 25  # None uses entire split
MAX_NEW_TOK = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_OUT = "boolq_smol_ctx.csv" if INCLUDE_PASSAGE else "boolq_smol_q.csv"
ENERGY_OUT = "Energy"

# ─── quiet mode ───
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
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


def norm(txt: str) -> str:
    t = txt.strip().lower()
    return (
        "true"
        if any(w in t for w in YES)
        else "false" if any(w in t for w in NO) else t
    )


def make_prompt(in_prompt_text):
    q = in_prompt_text["question"]
    ctx = in_prompt_text.get("passage", "")
    if INCLUDE_PASSAGE:
        return f"### Instruction:\nAnswer true or false only by using the passage.\n\n### Passage:\n{ctx}\n\n### Question:\n{q}\n\n### Response:\n"
    return f"### Instruction:\nAnswer true or false only.\n\n### Question:\n{q}\n\n### Response:\n"


# ─── evaluation loop with progress bar ───
os.makedirs(ENERGY_OUT, exist_ok=True)
em_sum = energy_sum = 0.0
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f, tqdm(
    total=len(ds), desc="BoolQ eval", ncols=80
) as bar:  # tqdm usage :contentReference[oaicite:3]{index=3}
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
        pred, gold = norm(pred_raw), ("true" if ex["answer"] else "false")
        em = int(pred == gold)
        em_sum += em
        energy_sum += float(energy)
        wr.writerow([idx, pred_raw, gold, em, f"{energy:.6f}"])
        bar.set_postfix(acc=f"{em_sum/(idx+1):.3f}")  # live acc on bar
        bar.update()

# ─── summary line ───
avg_em, avg_e = em_sum / len(ds), energy_sum / len(ds)
with open("avg_results.txt", "a", encoding="utf-8") as fp:
    fp.write(
        f"{DATASET_NAME}|{'q+r' if INCLUDE_PASSAGE else 'q'}|{MODEL_NAME}|{avg_em:.4f}|{avg_e:.6f}\n"
    )
print(f"Done → {CSV_OUT} | EM={avg_em:.4f} | kWh/qa={avg_e:.6f}")

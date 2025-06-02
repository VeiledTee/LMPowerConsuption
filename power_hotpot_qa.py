import csv
import re
import string
import time
from datetime import date

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── toggle here ───
INCLUDE_PASSAGE = False  # True → add supporting paragraphs
# ───────────────────

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "hotpotqa/hotpot_qa"
SPLIT = "validation"
N_SAMPLES = 5  # None → full split
MAX_NEW_TOK = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_OUT = (
    "hotpot_smol_q_ctx_results.csv"
    if INCLUDE_PASSAGE
    else "hotpot_smol_q_only_results.csv"
)


# ─── HotpotQA‑official normaliser & scorers ───
def normalize_answer(s: str) -> str:
    """Official HotpotQA answer normalisation."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / max(len(pred_tokens), 1)
    rec = len(common) / max(len(gold_tokens), 1)
    return 2 * prec * rec / (prec + rec)


# ─── load model & data ───
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
    )
    .to(DEVICE)
    .eval()
)

ds = load_dataset(DATASET_NAME, "fullwiki", split=SPLIT, trust_remote_code=True)
if N_SAMPLES:
    ds = ds.select(range(N_SAMPLES))


def make_prompt(in_prompt_text):
    q = in_prompt_text["question"]
    if not INCLUDE_PASSAGE:
        return (
            f"### Instruction:\nAnswer the question briefly and factually.\n\n"
            f"### Question:\n{q}\n\n### Response:\n"
        )
    # build context from supporting titles
    titles = {t for t, _ in in_prompt_text["supporting_facts"]}
    ctx = "\n\n".join(
        " ".join(sentence) for t, sentence in in_prompt_text["context"] if t in titles
    )
    if not ctx:
        ctx = "Context unavailable."
    return (
        f"### Instruction:\nAnswer the question briefly and factually using the context.\n\n"
        f"### Context:\n{ctx}\n\n### Question:\n{q}\n\n### Response:\n"
    )


# ─── evaluation loop ───
energy_sum = 0.0
t0 = time.perf_counter()

with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_file:
    wr = csv.writer(csv_file)
    wr.writerow(["qid", "predicted", "gold", "em", "f1", "energy_kWh"])

    em_sum = f1_sum = 0.0
    for idx, ex in enumerate(ds):
        tracker = EmissionsTracker(
            project_name=(
                "hotpot_smol_ctx" if INCLUDE_PASSAGE else "hotpot_smol_q_only"
            ),
            log_level="error",
            output_dir=".",
            output_file=f"Energy/energy_{idx}.csv",
        )
        tracker.start()
        with torch.inference_mode():
            out = model.generate(
                **tok(mak, return_tensors="pt").to(DEVICE),
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
            )
        energy = tracker.stop()  # kWh for this Q
        energy_sum += float(energy)

        pred = (
            tok.decode(out[0], skip_special_tokens=True)
            .split("### Response:")[-1]
            .strip()
        )
        gold = ex["answer"]

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        em_sum += em
        f1_sum += f1

        wr.writerow([idx, pred, gold, em, f1, f"{energy:.6f}"])

avg_em = em_sum / len(ds)
avg_f1 = f1_sum / len(ds)
avg_energy = energy_sum / len(ds)
t_total = time.perf_counter() - t0
today = date.today().isoformat()

# ─── summary line ───
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

#!/usr/bin/env python3
import csv
import logging
import time
import warnings
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ── static hyper‑params ───────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "google/boolq"
SPLIT = "validation"
N_SAMPLES = None
MAX_NEW_TOK = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
ENERGY_DIR = Path("Energy")
MODES = {"q+r": True}  # {"q": False, "q+r": True}

YES, NO = {"yes", "true"}, {"no", "false"}

# ── logging / warnings ────────────────────────────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)
ENERGY_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────
def norm(text: str) -> str:
    t = text.lower()
    if "true" in t or ("yes" in t and "no" not in t):
        return "true"
    if "false" in t or ("no" in t and "yes" not in t):
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


# ── evaluation wrapper (batched, resumable) ───────────────────────────
def run_mode(tag: str, include_passage: bool, dataset, model, tokenizer) -> None:
    csv_out = Path(f"boolq_smol_{tag}.csv")

    start_qid, mode = 0, "w"
    if csv_out.exists():
        last = ""
        with csv_out.open() as f:
            for line in f:
                if line.strip():
                    last = line
        if last and last.split(",")[0].isdigit():
            start_qid = int(last.split(",")[0]) + 1
            mode = "a"
    print(f"{tag}: resuming at qid {start_qid}")

    remaining = len(dataset) - start_qid
    with (
        csv_out.open(mode, newline="", encoding="utf-8") as f_out,
        tqdm(total=remaining, desc=f"Batches {tag}", ncols=80) as pbar,
    ):
        writer = csv.writer(f_out)
        if mode == "w":
            writer.writerow(
                ["qid", "raw_pred", "pred", "gold", "em", "energy_kWh", "time (s)"]
            )

        for batch_start in range(start_qid, len(dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            tracker = EmissionsTracker(
                project_name=f"boolq_{tag}",
                output_dir=str(ENERGY_DIR),
                output_file=f"energy_{tag}_{batch_start}_{batch_end - 1}.csv",
                log_level="error",
            )

            for qid, ex in enumerate(batch, start=batch_start):
                prompt = build_prompt(
                    ex["question"], ex.get("passage", ""), include_passage
                )
                t_0 = time.time()
                tracker.start()
                with torch.inference_mode():
                    out = model.generate(
                        **tokenizer(prompt, return_tensors="pt").to(DEVICE),
                        max_new_tokens=MAX_NEW_TOK,
                        do_sample=False,
                    )
                q_kwh = tracker.stop()
                elapsed = time.time() - t_0

                raw_pred = (
                    tokenizer.decode(out[0], skip_special_tokens=True)
                    .split("### Response:")[-1]
                    .strip()
                )
                pred = norm(raw_pred)
                gold = "true" if ex["answer"] else "false"
                em = int(pred == gold)

                writer.writerow([qid, raw_pred, pred, gold, em, q_kwh, elapsed])
                f_out.flush()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            pbar.update(len(batch))

    print(f"{tag}: finished; results saved to {csv_out}")


# ─── main: load resources once, run modes ─────────────────────────────
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
        )
        .to(DEVICE)
        .eval()
    )
    data = load_dataset(DATASET_NAME, split=SPLIT)
    if N_SAMPLES:
        data = data.select(range(N_SAMPLES))

    for tag, ctx in MODES.items():
        run_mode(tag, ctx, data, mdl, tok)
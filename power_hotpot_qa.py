import csv
import logging
import re
import string
import time
import warnings
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ─── fixed hyper‑params ────────────────────────────────────────────────
MODEL_NAME = "distilbert/distilgpt2"
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
N_SAMPLES = None
MAX_NEW_TOK = 64
BATCH_SIZE = 128
DEVICE = "cpu"
MODES = {"q": False, "q+r": True}  # {"q": False, "q+r": True}

ENERGY_DIR = Path("Energy").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# ─── quiet library chatter ────────────────────────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)


# ─── token normaliser & per‑row scorers (kept) ────────────────────────
def _normalize(txt: str) -> str:
    txt = txt.lower()
    txt = "".join(ch for ch in txt if ch not in string.punctuation)
    txt = re.sub(r"\b(a|an|the)\b", " ", txt)
    return " ".join(txt.split())


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pt, gt = _normalize(pred).split(), _normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)


def build_prompt(ex: dict, include_passage: bool) -> str:
    q = ex["question"]
    if not include_passage:
        return (
            "### Instruction:\nAnswer briefly and factually.\n\n"
            f"### Question:\n{q}\n\n### Response:\n"
        )
    titles = {t for t in ex["supporting_facts"]["title"]}
    context = ""
    if titles != {}:
        context = ". ".join(ex["context"]["title"])
        for s in ex["context"]["sentences"]:
            context += "".join(s)

    return (
        "### Instruction:\nAnswer using the context.\n\n"
        f"### Context:\n{context}\n\n### Question:\n{q}\n\n### Response:\n"
    )


# ─── single‑mode runner (CSV only) ────────────────────────────────────
def run_mode(tag: str, include_passage: bool, dataset, model, tokenizer) -> None:
    csv_out = Path(f"hotpot_smol_{tag}.csv")

    start_qid, mode = 0, "w"
    if csv_out.exists():
        last = ""
        with csv_out.open() as f:
            for l in f:
                if l.strip():
                    last = l
        if last and last.split(",")[0].isdigit():
            start_qid = int(last.split(",")[0]) + 1
            mode = "a"
    print(f"{tag}: resuming at qid {start_qid}")

    remaining = len(dataset) - start_qid
    with csv_out.open(mode, newline="", encoding="utf-8") as fout, tqdm(
        total=remaining, desc=f"Batches {tag}", ncols=80
    ) as pbar:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(
                ["qid", "pred", "gold", "em", "f1", "energy_kWh", "time (s)"]
            )

        for batch_start in range(start_qid, len(dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            tracker = EmissionsTracker(
                project_name=f"hotpot_{tag}",
                output_dir=str(ENERGY_DIR),
                output_file=f"energy_{tag}_{batch_start}_{batch_end-1}.csv",
                log_level="error",
            )

            for qid, ex in enumerate(batch, start=batch_start):
                prompt = build_prompt(ex, include_passage)
                t0 = time.time()
                tracker.start()
                with torch.inference_mode():
                    try:
                        out = model.generate(
                        **tokenizer(prompt, return_tensors="pt").to(DEVICE),
                        max_new_tokens=MAX_NEW_TOK,
                        do_sample=False,
                        )
                    except torch.OutOfMemoryError:
                        out = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=None).to('cpu').eval().generate(
                            **tokenizer(prompt, return_tensors="pt").to('cpu'),
                            max_new_tokens=MAX_NEW_TOK,
                            do_sample=False,
                        )
                kwh = tracker.stop()
                elapsed = time.time() - t0

                pred = (
                    tokenizer.decode(out[0], skip_special_tokens=True)
                    .split("### Response:")[-1]
                    .strip()
                )
                gold = ex["answer"]
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)

                writer.writerow([qid, pred, gold, em, f1, kwh, elapsed])
                fout.flush()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            pbar.update(len(batch))

    print(f"{tag}: finished; results saved to {csv_out}")


# ─── main: load once, run modes ───────────────────────────────────────
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else None
        )
        .to(DEVICE)
        .eval()
    )
    data = load_dataset(DATASET_NAME, CONFIG, split=SPLIT, trust_remote_code=True)
    if N_SAMPLES:
        data = data.select(range(N_SAMPLES))

    for tag, ctx in MODES.items():
        run_mode(tag, ctx, data, mdl, tok)

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
MODEL_NAME = "distilbert/distilgpt2"  # openai-community/gpt2-xl OR distilbert/distilgpt2
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
N_SAMPLES = None
MAX_NEW_TOK = 64
BATCH_SIZE = 128
DEVICE = "cpu"
MODES = {"q": False, "q+r": True}  # {"q": False, "q+r": True}
print(f"{'='*25}\nMODEL: {MODEL_NAME}\nMODES: {MODES}\n{'='*25}")

ENERGY_DIR = Path("Energy").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# MODEL_NAME = "openai-community/gpt2-xl"  # openai-community/gpt2-xl OR distilbert/distilgpt2
# DATASET_NAME = "hotpotqa/hotpot_qa"
# CONFIG = "fullwiki"
# SPLIT = "validation"
# N_SAMPLES = None
# MAX_NEW_TOK = 64
# BATCH_SIZE = 32
# DEVICE = "cpu"
# MODES = {"q": False}  # {"q": False, "q+r": True}
# print(f"{'='*25}\nMODEL: {MODEL_NAME}\nMODES: {MODES}\n{'='*25}")

ENERGY_DIR = Path("Energy").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# ─── quiet library chatter ────────────────────────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)


# ─── token normaliser & per‑row scorers ────────────────────────
def parse_last_cc_row(cc_file: Path) -> dict:
    # CodeCarbon writes one header + one data row in our use‑case
    with cc_file.open() as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    return dict(zip(header, values))


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
        return f"Question: {q}\nAnswer:"
    titles = {t for t in ex["supporting_facts"]["title"]}
    context = ""
    if titles:
        context = ". ".join(ex["context"]["title"])
        for s in ex["context"]["sentences"]:
            context += "".join(s)
    return f"Context: {context}\nQuestion: {q}\nAnswer:"


# ─── single‑mode runner (CSV only) ────────────────────────────────────
def run_mode(tag: str, include_passage: bool, dataset, model, tokenizer) -> None:
    csv_out = Path(f"hotpot_{MODEL_NAME.split('/')[-1]}_{tag}.csv")

    start_qid, mode = 0, "w"
    if csv_out.exists():
        last = ""
        with csv_out.open(encoding="utf-8") as f:
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
                ["qid", "pred", "gold", "em", "f1", "energy_kWh", "emissions (kg)", "time (s)"]
            )

        for batch_start in range(start_qid, len(dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))
            cc_outfile = f"energy_{MODEL_NAME.split('/')[-1]}_{tag}_{batch_start}_{batch_end-1}.csv"

            tracker = EmissionsTracker(
                project_name=f"hotpot_{MODEL_NAME.split('/')[-1]}_{tag}",
                output_dir=str(ENERGY_DIR),
                output_file=cc_outfile,
                log_level="error",
            )

            for qid, ex in enumerate(batch, start=batch_start):
                prompt = build_prompt(ex, include_passage)
                tracker.start()
                with torch.inference_mode():
                    try:
                        out = model.generate(
                            **tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_NEW_TOK).to(DEVICE),
                            max_new_tokens=MAX_NEW_TOK,
                            do_sample=False,
                        )
                    except torch.OutOfMemoryError:
                        out = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=None).to('cpu').eval().generate(
                            **tokenizer(prompt, return_tensors="pt").to('cpu'),
                            max_new_tokens=MAX_NEW_TOK,
                            do_sample=False,
                        )
                tracker.stop()
                row = parse_last_cc_row(Path(f"Energy/{cc_outfile}"))

                pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()

                gold = ex["answer"]
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)
                kwh = row["energy_consumed"]
                emissions_kg = row["emissions"]
                print(f"qid: {qid}\nPRED:\n{pred}\nGOLD:\n{gold}\n{'-' * 50}")
                writer.writerow([qid, pred, gold, em, f1, kwh, emissions_kg, row["duration"]])
                fout.flush()

            # torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()
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

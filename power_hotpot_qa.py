import csv
import logging
import re
import string
import time
import warnings
from datetime import date
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_log

# ─── fixed hyper‑params ────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "fullwiki"
SPLIT = "validation"
N_SAMPLES = None  # None → full split
MAX_NEW_TOK = 64
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODES = {"q": False, "q+r": True}  # tag → include_passage flag
ENERGY_DIR = Path("Energy").resolve()
RESULTS_TXT = Path("avg_results.txt").resolve()
ENERGY_DIR.mkdir(exist_ok=True)

# ─── silence transformers / codecarbon chatter ────────────────────────
warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("codecarbon").setLevel(logging.ERROR)


# ─── HotpotQA helpers ─────────────────────────────────────────────────
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
    titles = {t for t, _ in ex["supporting_facts"]}
    ctx = (
        "\n\n".join(" ".join(s) for t, s in ex["context"] if t in titles)
        or "Context unavailable."
    )
    return (
        "### Instruction:\nAnswer using the context.\n\n"
        f"### Context:\n{ctx}\n\n### Question:\n{q}\n\n### Response:\n"
    )


# ─── single‑mode runner ───────────────────────────────────────────────
def run_mode(tag: str, include_passage: bool, dataset, model, tokenizer) -> None:
    csv_out = Path(f"hotpot_smol_{tag}.csv")

    # resume point
    start_qid, mode = 0, "w"
    if csv_out.exists():
        last_line = ""
        with csv_out.open() as f:
            for l in f:
                if l.strip():
                    last_line = l
        if last_line and last_line.split(",")[0].isdigit():
            start_qid = int(last_line.split(",")[0]) + 1
            mode = "a"
    print(f"{tag}: resuming at qid {start_qid}")

    em_sum = f1_sum = energy_sum = 0.0
    t0 = time.perf_counter()

    with csv_out.open(mode, newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(["qid", "pred", "gold", "em", "f1", "energy_kWh"])

        for batch_start in range(start_qid, len(dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            tracker = EmissionsTracker(
                project_name=f"hotpot_{tag}",
                output_dir=str(ENERGY_DIR),
                output_file=f"energy_{tag}_{batch_start}_{batch_end-1}.csv",
                log_level="error",
            ).start()

            rows = []
            for ex in batch:
                prompt = build_prompt(ex, include_passage)
                with torch.inference_mode():
                    out = model.generate(
                        **tokenizer(prompt, return_tensors="pt").to(DEVICE),
                        max_new_tokens=MAX_NEW_TOK,
                        do_sample=False,
                    )
                pred = (
                    tokenizer.decode(out[0], skip_special_tokens=True)
                    .split("### Response:")[-1]
                    .strip()
                )
                gold = ex["answer"]
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)

                em_sum += em
                f1_sum += f1
                rows.append([ex["idx"], pred, gold, em, f1])

            batch_kwh = tracker.stop()
            energy_sum += batch_kwh
            kwh_per_q = batch_kwh / len(rows)

            for r in rows:
                writer.writerow([*r, f"{kwh_per_q:.6f}"])
            fout.flush()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # write summary
    n_done = len(dataset) - start_qid
    avg_em = em_sum / n_done
    avg_f1 = f1_sum / n_done
    avg_energy = energy_sum / n_done
    wall = time.perf_counter() - t0
    today = date.today().isoformat()

    with RESULTS_TXT.open("a", encoding="utf-8") as fp:
        fp.write(
            f"{today}|{DATASET_NAME}|{tag}|{MODEL_NAME}|{n_done}|"
            f"{avg_em:.4f}|{avg_f1:.4f}|{avg_energy:.6f}|{wall:.2f}\n"
        )

    print(
        f"{tag}: EM={avg_em:.4f} | F1={avg_f1:.4f} | "
        f"kWh/qa={avg_energy:.6f} | s={wall:.2f}"
    )


# ─── main: load resources once, run both modes ────────────────────────
if __name__ == "__main__":
    # load once
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

import csv, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker

# ─── toggle here ───
INCLUDE_PASSAGE = False  # True → add supporting paragraphs
# ───────────────────

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "google/boolq"
SPLIT = "validation"
N_SAMPLES = 200  # None → full split
MAX_NEW_TOK = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_OUT = (
    "boolq_smol_q_passage_results.csv"
    if INCLUDE_PASSAGE
    else "boolq_smol_q_only_results.csv"
)

# ─── load model & data ─────────────────────────────────
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

# ─── helpers ───────────────────────────────────────────
YES = {"yes", "true"}
NO = {"no", "false"}


def normalize_bool(text: str) -> str:
    text = text.strip().lower()
    if any(t in text for t in YES):
        return "true"
    if any(t in text for t in NO):
        return "false"
    return text  # fallback


def build_prompt(ex):
    q = ex["question"]
    if not INCLUDE_PASSAGE:
        return (
            f"### Instruction:\nAnswer true or false only.\n\n"
            f"### Question:\n{q}\n\n### Response:\n"
        )
    ctx = ex["passage"]
    return (
        f"### Instruction:\nAnswer true or false only by using the passage.\n\n"
        f"### Passage:\n{ctx}\n\n"
        f"### Question:\n{q}\n\n### Response:\n"
    )


# ─── evaluation loop ──────────────────────────────────
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["qid", "predicted", "gold", "em", "energy_kWh"])

    em_sum = 0
    energy_sum = 0
    for idx, ex in enumerate(ds):
        prompt = build_prompt(ex)

        tracker = EmissionsTracker(
            project_name=("boolq_smol_ctx" if INCLUDE_PASSAGE else "boolq_smol_q_only"),
            log_level="error",
            output_dir=".",
            output_file=f"Energy/energy_{idx}.csv",
        )
        tracker.start()
        with torch.inference_mode():
            out = model.generate(
                **tok(prompt, return_tensors="pt").to(DEVICE),
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
                temperature=0.0,
            )
        energy = tracker.stop()

        pred_raw = (
            tok.decode(out[0], skip_special_tokens=True)
            .split("### Response:")[-1]
            .strip()
        )
        pred = normalize_bool(pred_raw)
        gold = "true" if ex["answer"] else "false"

        em = int(pred == gold)
        em_sum += em
        energy_sum += energy

        wr.writerow([idx, pred_raw, gold, em, f"{energy:.6f}"])
        running_acc = em_sum / (idx + 1)
        print(
            f"{idx:>3}  EM={em}  running_acc={running_acc:.3f}  energy={energy:.6f} kWh"
        )

avg_em      = em_sum / len(ds)
avg_energy  = energy_sum / len(ds)  # make sure you accumulated energy_sum

line = f"{DATASET_NAME} | {'Question + Passage' if INCLUDE_PASSAGE else 'Question'} | {MODEL_NAME} | {avg_em:.4f} | {avg_energy:.6f}\n"

with open("avg_results.txt", "a", encoding="utf-8") as fp:
    fp.write(line)

print(f"Appended to avg_results.txt:\n{line.strip()}")
print(f"\nFinished → {CSV_OUT}")

import csv, re, string, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker

# ─── toggle here ───
INCLUDE_PASSAGE = False  # True → add supporting paragraphs
# ───────────────────

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET_NAME = "hotpotqa/hotpot_qa"
SPLIT = "validation"
N_SAMPLES = 25  # None → full split
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


def build_prompt(ex):
    q = ex["question"]
    if not INCLUDE_PASSAGE:
        return (
            f"### Instruction:\nAnswer the question briefly and factually.\n\n"
            f"### Question:\n{q}\n\n### Response:\n"
        )
    # build context from supporting titles
    titles = {t for t, _ in ex["supporting_facts"]}
    ctx = "\n\n".join(" ".join(sents) for t, sents in ex["context"] if t in titles)
    if not ctx:
        ctx = "Context unavailable."
    return (
        f"### Instruction:\nAnswer the question briefly and factually using the context.\n\n"
        f"### Context:\n{ctx}\n\n### Question:\n{q}\n\n### Response:\n"
    )


# ─── evaluation loop ───
with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_file:
    wr = csv.writer(csv_file)
    wr.writerow(["qid", "predicted", "gold", "em", "f1", "energy_kWh"])

    em_sum = f1_sum = 0.0
    for idx, ex in enumerate(ds):
        prompt = build_prompt(ex)
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
                **tok(prompt, return_tensors="pt").to(DEVICE),
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
                temperature=0.0,
            )
        energy = tracker.stop()  # kWh for this question
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
        print(f"{idx:>3}  EM={em}  F1={f1:.2f}  energy={energy:.6f} kWh")

avg_em = em_sum / len(ds)
avg_f1 = f1_sum / len(ds)
print(f"\nFinished → {CSV_OUT}")
print(f"Average EM = {avg_em:.3f} | Average F1 = {avg_f1:.3f}")

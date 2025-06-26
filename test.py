import re

import pandas as pd

file_path = "results/boolq_128_gemma-7b-it_q_simplified.csv"
df = pd.read_csv(file_path)


def normalize(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def exact_match(pred, gold) -> int:
    return int(normalize(pred) == normalize(gold))


def f1_score(pred, gold) -> float:
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)


# Recalculate columns
df["em"] = df.apply(lambda r: exact_match(r["pred"], r["gold"]), axis=1)
df["f1"] = df.apply(lambda r: f1_score(r["pred"], r["gold"]), axis=1)

# Save updated DataFrame
output_path = "boolq_128_gemma-7b-it_q_recalculated.csv"
df.to_csv(output_path, index=False)

from utils import normalize
from collections import Counter


def exact_match(pred: str, gold: str) -> int:
    """
    HotpotQA-style exact match: case/punctuation/articles-insensitive.
    """
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    HotpotQA-style token-level F1.
    """
    norm_pred = normalize(pred)
    norm_gold = normalize(gold)

    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
        return 0.0
    if norm_gold in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
        return 0.0

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec = num_same / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)

from collections import Counter
from src.utils import normalize

def exact_match(pred: str, gold: str) -> int:
    """
    2WikiMultiHopQA-style exact match: normalized, case/punctuation/articles-insensitive.
    """
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    2WikiMultiHopQA-style token-level F1.
    """
    norm_pred = normalize(pred)
    norm_gold = normalize(gold)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec = num_same / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)

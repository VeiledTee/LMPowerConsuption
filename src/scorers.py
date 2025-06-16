import re

from utils import normalize


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)

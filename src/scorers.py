from utils import normalize


def exact_match(pred: str, gold: str) -> int:
    """
    Compute exact match score between predicted and gold strings after normalization.

    Args:
        pred (str): Predicted answer.
        gold (str): Ground-truth answer.

    Returns:
        int: 1 if normalized strings match exactly, else 0.
    """
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    Compute token-level F1 score between predicted and gold answers after normalization.

    Args:
        pred (str): Predicted answer.
        gold (str): Ground-truth answer.

    Returns:
        float: F1 score (harmonic mean of precision and recall).
    """
    pt, gt = normalize(pred).split(), normalize(gold).split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec = len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)

from typing import Tuple


def prf_from_counts(tp, fp, fn) -> Tuple[float, float, float]:
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)
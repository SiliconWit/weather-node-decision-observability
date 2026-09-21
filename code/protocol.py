"""The evaluation protocol: blocked splits, scaling, the learner, and the scores.

Every experiment in the study goes through score_head, so that a difference between
two numbers is a difference in the inputs and never a difference in how they were
scored.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, brier_score_loss

FOLDS, EMBARGO, C_REG = 5, 30, 100.0
HEADS = ["irrigation_timing", "fertilizer_timing", "pest_risk"]


def blocked_splits(n, n_folds=FOLDS, embargo=EMBARGO):
    """Yield (train, test) boolean masks over n consecutive days.

    The test set is one contiguous block. Training excludes the test block and
    `embargo` days on each side of it, so that a rolling mean on a training day
    cannot contain a test day.
    """
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        test = np.zeros(n, bool); test[lo:hi] = True
        train = np.ones(n, bool)
        train[max(0, lo - embargo):min(n, hi + embargo)] = False
        yield train, test


def scaled(Xtr, Xte):
    """Min-max to [-1, 1), fitted on the training rows only.

    The upper end is 1 - 2**-15 rather than 1, because that is the largest value a
    Q15 register holds. Fitting the scaler on all rows is a leak.
    """
    lo, hi = Xtr.min(0), Xtr.max(0)
    rg = np.maximum(hi - lo, 1e-9)
    u = lambda Z: np.clip(2 * (Z - lo) / rg - 1, -1, 1 - 2 ** -15)
    return u(Xtr), u(Xte)


def fit_lr(Xtr, ytr):
    """Logistic regression, returned as (w, b) rescaled so every coefficient fits Q15.

    The decision is the sign of the affine score. Multiplying w and b by the same
    positive constant cannot change a sign, so the rescaling changes no decision.
    """
    lr = LogisticRegression(max_iter=8000, C=C_REG).fit(Xtr, ytr)
    w, b = lr.coef_[0].copy(), float(lr.intercept_[0])
    s = 0.95 / max(np.abs(w).max(), abs(b))
    return w * s, b * s


def score_head(X, y, ceiling=False):
    """Cross-validated accuracy, Matthews correlation, Brier score and majority rate.

    Returns {metric: (mean, sd)} over the folds, plus the fold count. With
    ceiling=True a gradient-boosted model is fitted on the same folds as a capacity
    reference. It is not a guaranteed upper bound.
    Folds whose training or test block contains a single class are skipped.
    """
    out = {k: [] for k in ("acc", "mcc", "brier", "maj", "ceil")}
    for tr, te in blocked_splits(len(y)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xtr, Xte = scaled(X[tr], X[te])
        w, b = fit_lr(Xtr, y[tr])
        s = Xte @ w + b
        yh = (s > 0).astype(int)
        out["acc"].append(float((yh == y[te]).mean()))
        out["mcc"].append(float(matthews_corrcoef(y[te], yh)))
        out["brier"].append(float(brier_score_loss(y[te], 1 / (1 + np.exp(-s)))))
        out["maj"].append(float((np.bincount(y[tr]).argmax() == y[te]).mean()))
        if ceiling:
            gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, y[tr])
            out["ceil"].append(float(matthews_corrcoef(y[te], gb.predict(Xte))))
    r = {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items() if v}
    r["folds"] = len(out["mcc"])
    return r

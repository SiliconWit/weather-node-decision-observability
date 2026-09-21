"""The split and the learner. The first test catches the error that invalidates
every later number."""
import numpy as np

import protocol as P


def test_test_blocks_partition_the_record_and_training_respects_the_embargo():
    n = 5200
    seen = np.zeros(n, int)
    for tr, te in P.blocked_splits(n, 5, 30):
        idx = np.flatnonzero(te)
        assert np.all(np.diff(idx) == 1), "each test block must be contiguous in time"
        seen += te
        assert not np.any(tr & te)
        gap = np.min(np.abs(np.flatnonzero(tr)[:, None] - idx[None, [0, -1]]))
        assert gap > 30, "a training day sits inside the embargo around the test block"
    assert np.all(seen == 1), "every day must be tested exactly once"


def test_scaler_is_fitted_on_training_rows_only():
    Xtr = np.array([[0.0], [10.0]])
    Xte = np.array([[20.0]])
    a, b = P.scaled(Xtr, Xte)
    assert a.min() == -1.0
    assert b.max() < 1.0, "values beyond the training range must be clipped below 1"


def test_coefficient_rescaling_changes_no_decision():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (400, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(400) > 0).astype(int)
    w, b = P.fit_lr(X, y)
    assert max(np.abs(w).max(), abs(b)) <= 0.95 + 1e-12
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=8000, C=P.C_REG).fit(X, y)
    assert np.array_equal(lr.predict(X), (X @ w + b > 0).astype(int))

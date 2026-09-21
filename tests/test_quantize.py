"""Fixed point: the bound must hold, whatever the data."""
import numpy as np

import quantize as Q


def test_quantise_saturates_to_the_register():
    q = Q.quantise(np.array([-3.0, 0.0, 3.0]), 15)
    assert q.min() == -1.0 and q.max() == 1.0 - 2.0 ** -15


def test_measured_flips_never_exceed_the_bound():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1 - 2 ** -15, (2000, 12))
    w = rng.uniform(-0.9, 0.9, 12); b = 0.05
    for bits in range(3, 17):
        assert Q.flip_rate_measured(X, w, b, bits) <= Q.flip_rate_bound(X, w, b, bits) + 1e-12

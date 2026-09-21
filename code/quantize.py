"""Fixed-point deployment of a linear score, and the flip-rate bound.

A logistic regression decides by the sign of an affine score. The sigmoid is
monotone, so it cannot change a hard decision: only the score's sign matters.
That reduces the whole question of deployment error to a margin condition.
"""
import numpy as np


def q_step(frac_bits):
    """Quantisation step of a QN fixed-point format."""
    return 2.0 ** (-frac_bits)


def quantise(v, frac_bits, saturate=True):
    """Round-to-nearest into QN, with saturation to the representable range."""
    q = q_step(frac_bits)
    out = np.round(np.asarray(v, float) / q) * q
    if saturate:
        out = np.clip(out, -1.0, 1.0 - q)
    return out


def score_float(X, w, b):
    return X @ w + b


def score_fixed(X, w, b, frac_bits, acc_bits=None):
    """Emulate the firmware: quantise inputs and coefficients, accumulate in a
    wide accumulator, then round the accumulator to acc_bits if given."""
    Xq = quantise(X, frac_bits)
    wq = quantise(w, frac_bits)
    bq = quantise(np.array([b]), frac_bits)[0]
    s = Xq @ wq + bq
    if acc_bits is not None:
        s = np.round(s / q_step(acc_bits)) * q_step(acc_bits)
    return s


def worst_case_perturbation(X, w, frac_bits):
    """Deterministic bound on |s_fixed - s_float|, per sample.

    s~ - s = w.ex + ew.x + ew.ex + eb, with every error at most q/2, so
    |s~ - s| <= (q/2)(||w||_1 + ||x||_1 + 1) + (q^2/4) d.
    """
    q = q_step(frac_bits)
    d = X.shape[1]
    return (q / 2.0) * (np.abs(w).sum() + np.abs(X).sum(axis=1) + 1.0) + (q * q / 4.0) * d


def gaussian_perturbation_sd(X, w, frac_bits):
    """Standard deviation of the perturbation when the rounding errors are
    treated as independent and uniform on [-q/2, q/2], whose variance is q^2/12."""
    q = q_step(frac_bits)
    var = (q * q / 12.0) * (np.sum(w ** 2) + np.sum(X ** 2, axis=1) + 1.0)
    return np.sqrt(var)


def flip_rate_bound(X, w, b, frac_bits):
    """Upper bound on the fraction of decisions that can change: a sample can
    only flip if its margin is no larger than the worst-case perturbation."""
    margin = np.abs(score_float(X, w, b))
    return float(np.mean(margin <= worst_case_perturbation(X, w, frac_bits)))


def flip_rate_estimate(X, w, b, frac_bits):
    """Expected flip rate under the independent-uniform-error model."""
    from scipy.stats import norm
    margin = np.abs(score_float(X, w, b))
    sd = gaussian_perturbation_sd(X, w, frac_bits)
    return float(np.mean(norm.cdf(-margin / np.maximum(sd, 1e-15))))


def flip_rate_measured(X, w, b, frac_bits):
    """The rate actually observed when the fixed-point path is emulated."""
    a = score_float(X, w, b) > 0
    c = score_fixed(X, w, b, frac_bits) > 0
    return float(np.mean(a != c))

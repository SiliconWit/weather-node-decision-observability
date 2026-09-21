"""A Richardson-type stochastic weather generator fitted to a daily record.

Structure: a two-state Markov chain for rain occurrence with seasonally varying
transition probabilities, a gamma distribution for wet-day amounts, and a
multivariate AR(1) process for the standardised residuals of the other variables
so that their cross-correlations and day-to-day persistence survive.

Reference: Richardson (1981), "Stochastic simulation of daily precipitation,
temperature, and solar radiation", Water Resources Research 17(1), 182-190.
"""
import numpy as np

VARS = ("T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN")
NH = 2          # annual plus semiannual, enough for temperature and humidity
NH_RAIN = 4     # rainfall needs more: the long and short rains make it bimodal
WINDOW = 15     # +/- days used to smooth the occurrence probabilities


def _design(doy, nh=NH):
    """Harmonic design matrix: constant plus nh sine/cosine pairs."""
    w = 2 * np.pi * doy / 365.25
    cols = [np.ones_like(doy, float)]
    for k in range(1, nh + 1):
        cols += [np.sin(k * w), np.cos(k * w)]
    return np.column_stack(cols)


def _fit_harmonic(doy, y, nh=NH):
    X = _design(doy, nh)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit(doy, data, rain, wet_thresh=1.0):
    """Fit the generator. `data` is a dict of arrays keyed by VARS."""
    m = {}
    wet = rain >= wet_thresh
    # Rain occurrence: P(wet|wet) and P(wet|dry), both seasonally varying.
    # Fitting harmonics straight onto the binary series extrapolates badly in the
    # dry season, where the conditioning subset is nearly empty. Estimating each
    # curve on a day-of-year window first, then fitting harmonics to that smooth
    # estimate, keeps the simulated wet-day fraction honest.
    prev = np.concatenate([[False], wet[:-1]])
    grid = np.arange(1, 366)
    pww = np.zeros(365); pwd = np.zeros(365)
    for i, d0 in enumerate(grid):
        near = np.minimum(np.abs(doy - d0), 365 - np.abs(doy - d0)) <= WINDOW
        a, b = near & prev, near & ~prev
        pww[i] = wet[a].mean() if a.sum() > 30 else wet[prev].mean()
        pwd[i] = wet[b].mean() if b.sum() > 30 else wet[~prev].mean()
    m["p_ww"] = _fit_harmonic(grid, pww, NH_RAIN)
    m["p_wd"] = _fit_harmonic(grid, pwd, NH_RAIN)
    # Wet-day amounts: gamma with a seasonal mean and a constant shape, fitted to
    # the excess over the threshold rather than the total. Fitting the total lets
    # the generator draw a "wet" day of 0.4 mm, which then counts as dry and drags
    # the simulated wet-day fraction well below the observed one.
    m["wet_thresh"] = wet_thresh
    amt = rain[wet] - wet_thresh
    m["rain_mu"] = _fit_harmonic(doy[wet], amt, NH_RAIN)
    mu = _design(doy[wet], NH_RAIN) @ m["rain_mu"]
    resid = amt / np.maximum(mu, 1e-6)
    m["rain_shape"] = max(0.3, (resid.mean() ** 2) / max(resid.var(), 1e-6))
    # other variables: seasonal mean and sd, then multivariate AR(1) on residuals
    Z = np.zeros((len(doy), len(VARS)))
    for j, v in enumerate(VARS):
        y = data[v]
        m[f"mu_{v}"] = _fit_harmonic(doy, y)
        r = y - _design(doy) @ m[f"mu_{v}"]
        m[f"sd_{v}"] = _fit_harmonic(doy, np.abs(r)) * np.sqrt(np.pi / 2)
        sd = np.maximum(_design(doy) @ m[f"sd_{v}"], 1e-6)
        Z[:, j] = r / sd
    M0 = np.cov(Z.T)
    M1 = (Z[:-1].T @ Z[1:]) / (len(Z) - 1)
    A = M1.T @ np.linalg.inv(M0)                 # AR(1) coefficient matrix
    S = M0 - A @ M1
    S = (S + S.T) / 2
    ev, V = np.linalg.eigh(S)
    m["A"], m["B"] = A, V @ np.diag(np.sqrt(np.clip(ev, 1e-12, None))) @ V.T
    return m


def simulate(m, n_days, rng, start_doy=1):
    """Generate n_days of weather. Returns a dict of arrays plus rain."""
    doy = ((np.arange(n_days) + start_doy - 1) % 365) + 1
    X = _design(doy)
    XR = _design(doy, NH_RAIN)
    p_ww = np.clip(XR @ m["p_ww"], 0.01, 0.99)
    p_wd = np.clip(XR @ m["p_wd"], 0.01, 0.99)
    rain = np.zeros(n_days); wet = False
    mu_r = np.maximum(XR @ m["rain_mu"], 0.1)
    k = m["rain_shape"]
    for i in range(n_days):
        wet = rng.random() < (p_ww[i] if wet else p_wd[i])
        rain[i] = m["wet_thresh"] + rng.gamma(k, mu_r[i] / k) if wet else 0.0
    z = np.zeros(len(VARS)); Z = np.zeros((n_days, len(VARS)))
    for i in range(n_days):
        z = m["A"] @ z + m["B"] @ rng.standard_normal(len(VARS))
        Z[i] = z
    out = {}
    for j, v in enumerate(VARS):
        mu = X @ m[f"mu_{v}"]
        sd = np.maximum(X @ m[f"sd_{v}"], 1e-6)
        out[v] = mu + sd * Z[:, j]
    out["RH2M"] = np.clip(out["RH2M"], 5, 100)
    out["WS2M"] = np.maximum(out["WS2M"], 0.1)
    out["ALLSKY_SFC_SW_DWN"] = np.maximum(out["ALLSKY_SFC_SW_DWN"], 0.5)
    out["T2M_MIN"] = np.minimum(out["T2M_MIN"], out["T2M_MAX"] - 0.5)
    out["PRECTOTCORR"] = rain
    out["doy"] = doy
    return out

"""Features: only what the node senses, and nothing from the future."""
import numpy as np

import features


def _sim(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {"T2M_MAX": 22 + rng.standard_normal(n), "T2M_MIN": 11 + rng.standard_normal(n),
            "RH2M": 75 + rng.standard_normal(n), "WS2M": 2 + rng.random(n),
            "PS": 79 + 0.1 * rng.standard_normal(n), "WD2M": 360 * rng.random(n),
            "PRECTOTCORR": 1000 + 100 * rng.random(n), "ALLSKY_SFC_SW_DWN": 500 + rng.random(n),
            "doy": (np.arange(n) % 365) + 1}


def test_no_rainfall_or_radiation_reaches_the_features():
    X, names = features.make_features(_sim())
    assert X.shape[1] == len(names)
    assert np.all(np.abs(X) < 900), (
        "a feature has the scale of the rainfall or radiation column; the node has "
        "neither a rain gauge nor a pyranometer")


def test_features_on_a_day_do_not_use_later_days():
    a = _sim()
    b = {k: v.copy() for k, v in a.items()}
    t = 120
    for k in ("T2M_MAX", "T2M_MIN", "RH2M", "PS", "WS2M"):
        b[k][t + 1:] += 5.0
    Xa, _ = features.make_features(a)
    Xb, _ = features.make_features(b)
    assert np.allclose(Xa[: t + 1], Xb[: t + 1]), "a rolling feature is looking ahead"


def test_calendar_columns_can_be_found():
    _, names = features.make_features(_sim())
    assert sorted(n for n in names if n.startswith("doy_")) == ["doy_cos", "doy_sin"]

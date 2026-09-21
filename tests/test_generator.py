"""The weather generator, fitted to a synthetic record whose statistics are known.

The real record is not needed, so these run anywhere. What they check is that
fitting and simulating preserve what matters downstream: how often it rains, the
seasonal mean temperature, and day-to-day persistence.
"""
import numpy as np

import weathergen as W


def _synthetic(n_years=20, seed=0):
    rng = np.random.default_rng(seed)
    n = 365 * n_years
    doy = (np.arange(n) % 365) + 1
    w = 2 * np.pi * doy / 365.25
    data = {}
    base = {"T2M_MAX": (23 + 2 * np.sin(w), 1.2), "T2M_MIN": (11 + np.cos(w), 1.0),
            "RH2M": (75 + 8 * np.sin(2 * w), 5.0), "WS2M": (2.3, 0.4),
            "PS": (79.0, 0.2), "ALLSKY_SFC_SW_DWN": (20 - 3 * np.cos(w), 2.5)}
    for v, (mu, sd) in base.items():
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = 0.7 * z[i - 1] + np.sqrt(1 - 0.49) * rng.standard_normal()
        data[v] = mu + sd * z
    wet = np.zeros(n, bool)
    for i in range(1, n):
        p = 0.6 if wet[i - 1] else 0.3
        wet[i] = rng.random() < p
    rain = np.where(wet, 1.0 + rng.gamma(0.8, 6.0, n), 0.0)
    return doy, data, rain


def test_simulated_weather_keeps_the_fitted_statistics():
    doy, data, rain = _synthetic()
    m = W.fit(doy, data, rain)
    sim = W.simulate(m, 365 * 20, np.random.default_rng(1))
    assert abs((sim["PRECTOTCORR"] >= 1.0).mean() - (rain >= 1.0).mean()) < 0.04, \
        "wet-day fraction drifted: check the occurrence chain and the wet-day threshold"
    assert abs(sim["T2M_MAX"].mean() - data["T2M_MAX"].mean()) < 0.3
    lag = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(lag(sim["T2M_MAX"]) - lag(data["T2M_MAX"])) < 0.1, \
        "persistence lost: the residual process must carry day-to-day memory"
    assert np.all(sim["T2M_MIN"] < sim["T2M_MAX"])

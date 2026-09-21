"""What the node can sense, and nothing else.

The node carries a temperature and humidity sensor, an anemometer and vane, and a
barometer. It has no rain gauge, no pyranometer and no soil moisture probe, and it
has a clock. Every feature must be computable on the node from those alone.
"""
import numpy as np


def make_features(sim, lags=(1, 3, 7, 14)):
    """Instantaneous readings, diurnal range, rolling means, tendencies, and the date.

    Returns (X, names). Rolling means are trailing, so a feature on day t uses
    nothing after day t. Day of year enters as a sine and cosine pair, named with
    the prefix `doy_`, so that the calendar ablation can find and remove them.

    Rainfall and radiation are in the simulated record. Neither may appear here.
    """
    base = {
        "tmax": sim["T2M_MAX"], "tmin": sim["T2M_MIN"],
        "tmean": (sim["T2M_MAX"] + sim["T2M_MIN"]) / 2,
        "trange": sim["T2M_MAX"] - sim["T2M_MIN"],
        "rh": sim["RH2M"], "u2": sim["WS2M"], "ps": sim["PS"],
        "wd_sin": np.sin(np.deg2rad(sim["WD2M"])),
        "wd_cos": np.cos(np.deg2rad(sim["WD2M"])),
    }
    cols, names = [], []
    for k, v in base.items():
        cols.append(v); names.append(k)
    for L in lags:
        for k in ("tmean", "trange", "rh", "ps"):
            v = base[k]
            c = np.convolve(v, np.ones(L) / L, mode="full")[:len(v)]
            cols.append(c); names.append(f"{k}_m{L}")
    for k in ("ps", "rh"):
        v = base[k]
        cols.append(np.concatenate([[0], np.diff(v)])); names.append(f"{k}_d1")
    cols.append(np.sin(2 * np.pi * sim["doy"] / 365.25)); names.append("doy_sin")
    cols.append(np.cos(2 * np.pi * sim["doy"] / 365.25)); names.append("doy_cos")
    return np.column_stack(cols), names

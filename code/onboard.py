"""The reference recursion as the node would run it, and how to score it.

Three pieces: a temperature-only estimate of evaporative demand, the water-balance
recursion driven by what the node has, and a scoring rule for events rather than days.
"""
import numpy as np
import reference as R
from eto import extraterrestrial


def eto_hargreaves(tmax, tmin, doy, lat_deg):
    """FAO-56 eq. 52: reference ETo from daily maximum and minimum temperature.

    Needs a thermometer and a clock, nothing else. Extraterrestrial radiation comes
    from the date and latitude, in MJ per square metre per day, and the 0.408 factor
    converts it to millimetres of water. run_onboard.py compares it with
    Penman-Monteith on the same days, because a small bias in a daily rate is not
    necessarily small in a quantity that accumulates it.
    """
    Ra = extraterrestrial(doy, lat_deg)
    tmean = (tmax + tmin) / 2.0
    return 0.0023 * 0.408 * Ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))


def onboard_decision(eto, rain, dap):
    """The same recursion the reference runs, as the node would run it.

    Follows reference.water_balance step for step, including the carry-over between
    seasons, and takes the decision from the post-update depletion, so that it is
    the identical comparison the reference policy makes. Given the reference's own
    inputs it reproduces the reference decisions exactly (tests/test_onboard.py).
    """
    n = len(eto)
    decide = np.zeros(n, int)
    D = 0.0
    for i in range(n):
        TAW = 1000.0 * (R.THETA_FC - R.THETA_WP) * R.root_depth(dap[i])
        etc_pot = R.kcb_curve(dap[i]) * eto[i]
        p = float(np.clip(R.P_TABLE + 0.04 * (5.0 - etc_pot), 0.1, 0.8))
        RAW = p * TAW
        ks = 1.0 if D <= RAW else max(0.0, (TAW - D) / (TAW - RAW))
        etc = ks * etc_pot
        irr = (RAW * 0.9) if D > RAW else 0.0
        ro = max(0.0, rain[i] - 0.2 * TAW)
        D = D - (rain[i] - ro) - irr + etc
        D = float(np.clip(D, 0.0, TAW))
        decide[i] = int(D > RAW)
    return decide


def event_match(pred, true, tol):
    """Precision, recall and F1 when a prediction within `tol` days counts as a hit.

    Returns a dict with keys recall, precision, f1. At tol=0 this is per-day
    scoring, under which an event predicted one day late counts as both a miss and
    a false alarm.
    """
    pi, ti = np.flatnonzero(pred), np.flatnonzero(true)
    if len(pi) == 0 or len(ti) == 0:
        return dict(recall=0.0, precision=0.0, f1=0.0)
    rec = sum(np.any(np.abs(pi - t) <= tol) for t in ti) / len(ti)
    pre = sum(np.any(np.abs(ti - p) <= tol) for p in pi) / len(pi)
    f1 = 0.0 if rec + pre == 0 else 2 * rec * pre / (rec + pre)
    return dict(recall=float(rec), precision=float(pre), f1=float(f1))

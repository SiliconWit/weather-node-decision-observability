"""The simulated record: fit the weather generator, simulate, run the reference policy.

    weather     NASA POWER daily record for the DeKUT grid cell, 2005 to 2024
    generator   fitted to that record, then run for YEARS simulated years
    policy      FAO-56 water balance, thermal time and Wallin severity, season by season

Everything later in the study reads from what this module returns, so a difference
here moves every number downstream.
"""
import json, os, datetime as dt
import numpy as np
from eto import eto_pm
from weathergen import fit, simulate, VARS
import reference as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

LAT, LON, ELEV = -0.4201, 36.9476, 2062.0   # POWER grid-cell elevation, not the town's 1795 m
YEARS, SEED, PLANT_DOY = 40, 1, 182


def load_observed(data_dir=DATA):
    """Read the cached POWER record. Returns (day of year, dict of variables, rainfall).

    The variable names are POWER's own. POWER marks missing values with -999; the
    cached record has none, and a record that does is rejected.
    """
    d = json.load(open(os.path.join(data_dir, "nyeri_daily.json")))["properties"]["parameter"]
    wd = json.load(open(os.path.join(data_dir, "nyeri_wd.json")))["properties"]["parameter"]["WD2M"]
    keys = sorted(d["T2M"].keys())
    dates = [dt.datetime.strptime(k, "%Y%m%d") for k in keys]
    g = lambda k: np.array([d[k][x] for x in keys], float)
    doy = np.array([x.timetuple().tm_yday for x in dates])
    data = {v: g(v) for v in VARS}
    data["WD2M"] = np.array([wd[x] for x in keys], float)
    rain = g("PRECTOTCORR")
    for k, v in list(data.items()) + [("PRECTOTCORR", rain)]:
        if np.any(v <= -990):
            raise ValueError(f"{k} contains POWER fill values (-999)")
    return doy, data, rain


def make_record(n_years=YEARS, seed=SEED, data_dir=DATA):
    """Fit the generator to the observed record and simulate n_years of daily weather.

    Wind direction is not part of the generator; it is drawn separately around the
    observed mean. Reference ETo is computed from the simulated weather with
    Penman-Monteith, and is the only evapotranspiration the reference policy sees.
    """
    doy_o, data_o, rain_o = load_observed(data_dir)
    m = fit(doy_o, data_o, rain_o)
    rng = np.random.default_rng(seed)
    sim = simulate(m, 365 * n_years, rng)
    mu = np.deg2rad(data_o["WD2M"].mean())
    kappa = 1.0 / max(np.var(np.deg2rad(data_o["WD2M"])), 1e-3)
    sim["WD2M"] = np.rad2deg(rng.vonmises(mu, min(kappa, 5.0), len(sim["doy"]))) % 360
    sim["ETO"] = eto_pm(sim["T2M_MAX"], sim["T2M_MIN"], sim["RH2M"], sim["WS2M"],
                        sim["ALLSKY_SFC_SW_DWN"], sim["PS"], sim["doy"], LAT, ELEV)
    return sim


def run_reference(sim, plant_doy=PLANT_DOY):
    """Season by season, run the physical model and read off the decisions.

    Returns (labels, latent, grow). `latent` holds the states each decision
    thresholds; `grow` marks the growing-season days that are evaluated.

    The planting day is a modelling choice. A long-rains crop is close to rain-fed
    and rarely needs irrigating; a dry-season crop, planted on day 182, is the case
    a decision-support node exists for, and is used throughout.
    """
    n = len(sim["doy"])
    season_len = sum(R.L.values())
    dap = np.zeros(n, int); starts = []
    for i in range(n - season_len):
        if sim["doy"][i] == plant_doy:
            starts.append(i)
            dap[i:i + season_len] = np.arange(1, season_len + 1)
    grow = dap > 0
    wb = R.water_balance(sim["ETO"], sim["PRECTOTCORR"], np.maximum(dap, 1))
    gdd = np.zeros(n); sv_acc = np.zeros(n)
    hh = np.clip(0.42 * (sim["RH2M"] - 55.0), 0, 24)
    sv = R.wallin_sv(hh, sim["T2M_MIN"] + 0.25 * (sim["T2M_MAX"] - sim["T2M_MIN"]))
    for s in starts:
        e = min(s + season_len, n)
        gdd[s:e] = R.thermal_time(sim["T2M_MAX"][s:e], sim["T2M_MIN"][s:e])
        sv_acc[s:e] = np.cumsum(sv[s:e])
    labels = {
        "irrigation_timing": (wb["Dr"] > wb["RAW"]).astype(int),
        "irrigation_depth": np.digitize(wb["Dr"], [10.0, 25.0]),
        "fertilizer_timing": ((gdd > 250) & (gdd < 400)).astype(int),
        "pest_risk": (sv_acc >= R.SV_SPRAY_THRESHOLD).astype(int),
    }
    latent = dict(Dr=wb["Dr"], RAW=wb["RAW"], Ks=wb["Ks"], gdd=gdd, sv_acc=sv_acc, dap=dap)
    return labels, latent, grow

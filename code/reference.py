"""The reference policy: an FAO-56 soil water balance plus thermal time and a
Wallin humid-hour blight model. Its decisions are the labels the node learns.

Every constant below is traceable. Sources are named beside each one.
Crop: potato, a major highland crop around Nyeri and the crop the Wallin blight
model was built for.
"""
import numpy as np

# --- soil: humic Nitisol, >35 percent clay. Closest FAO-56 Table 19 class is
# --- silty clay, theta_FC 0.35, theta_WP 0.23, TAW 120 mm per metre.
THETA_FC, THETA_WP = 0.35, 0.23

# --- crop: potato. FAO-56 Table 22 (Zr, p), Table 17 (Kcb), Table 11 (stages).
ZR_MAX, ZR_INI = 0.60, 0.25          # Table 22 gives 0.4 to 0.6 m
P_TABLE = 0.35                       # Table 22
KCB = {"ini": 0.15, "mid": 1.10, "end": 0.65}     # Table 17
L = {"ini": 25, "dev": 30, "mid": 45, "end": 30}  # Table 11, May planting, 130 d
TBASE, TUPPER = 7.0, 30.0            # potato cardinal temperatures

# --- Wallin (1962) severity values, as tabulated by UMaine Extension Bulletin
# --- 2418. Original bands are in Fahrenheit; the Celsius conversion is below.
#     45-54 F = 7.2-12.2 C : SV 0 below 15 h, then 16-18, 19-21, 22-24, 25-27
#     55-59 F = 12.8-15.0 C: SV 0 below 12 h, then 13-15, 16-18, 19-21, 22-24
#     60-81 F = 15.6-27.2 C: SV 0 below  9 h, then 10-12, 13-15, 16-18, 19-21
WALLIN_BANDS = [
    (7.2, 12.5, [15, 18, 21, 24]),
    (12.5, 15.3, [12, 15, 18, 21]),
    (15.3, 27.2, [9, 12, 15, 18]),
]
SV_SPRAY_THRESHOLD = 18              # accumulated SV after emergence


def kcb_curve(dap):
    """Basal crop coefficient over days after planting (FAO-56 figure 25)."""
    a = L["ini"]; b = a + L["dev"]; c = b + L["mid"]; d = c + L["end"]
    if dap <= a: return KCB["ini"]
    if dap <= b: return KCB["ini"] + (KCB["mid"] - KCB["ini"]) * (dap - a) / (b - a)
    if dap <= c: return KCB["mid"]
    if dap <= d: return KCB["mid"] + (KCB["end"] - KCB["mid"]) * (dap - c) / (d - c)
    return KCB["end"]


def root_depth(dap):
    frac = min(1.0, dap / (L["ini"] + L["dev"]))
    return ZR_INI + (ZR_MAX - ZR_INI) * frac


def water_balance(eto, rain, dap, irrigate=True):
    """Daily root-zone depletion, FAO-56 equations 82 to 85.

    Returns the latent state the node cannot observe.
    """
    n = len(eto)
    out = {k: np.zeros(n) for k in ("Dr", "Ks", "TAW", "RAW", "ETc", "irr", "DP")}
    D = 0.0
    for i in range(n):
        TAW = 1000.0 * (THETA_FC - THETA_WP) * root_depth(dap[i])      # eq 82
        etc_pot = kcb_curve(dap[i]) * eto[i]
        p = float(np.clip(P_TABLE + 0.04 * (5.0 - etc_pot), 0.1, 0.8))  # FAO-56 p adj
        RAW = p * TAW                                                   # eq 83
        ks = 1.0 if D <= RAW else max(0.0, (TAW - D) / (TAW - RAW))     # eq 84
        etc = ks * etc_pot
        irr = (RAW * 0.9) if (irrigate and D > RAW) else 0.0
        ro = max(0.0, rain[i] - 0.2 * TAW)
        D = D - (rain[i] - ro) - irr + etc                              # eq 85
        dp = max(0.0, -D)
        D = float(np.clip(D, 0.0, TAW))
        for k, v in zip(("Dr","Ks","TAW","RAW","ETc","irr","DP"),
                        (D, ks, TAW, RAW, etc, irr, dp)):
            out[k][i] = v
    return out


def thermal_time(tmax, tmin):
    """Accumulated growing degree days with capped cardinal temperatures."""
    tmx = np.clip(tmax, TBASE, TUPPER); tmn = np.clip(tmin, TBASE, TUPPER)
    return np.cumsum(np.maximum((tmx + tmn) / 2.0 - TBASE, 0.0))


def wallin_sv(humid_hours, t_during):
    """Wallin severity value per day from humid-period length and the mean
    temperature during that period."""
    sv = np.zeros(len(humid_hours))
    for i, (h, t) in enumerate(zip(humid_hours, t_during)):
        for lo, hi, cuts in WALLIN_BANDS:
            if lo <= t < hi:
                sv[i] = np.searchsorted(cuts, h, side="right")
                break
    return sv

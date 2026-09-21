"""Turn the results files into LaTeX macros, so no number is typed by hand.

    python3 code/build_numbers.py    # reads results/*.json; writes results/numbers.tex

Reads audit.json and onboard.json (the reported run, seed 1) and seed-spread.json
(the same experiments at every generator seed, from run_seeds.py). Each spread
macro is the smallest and largest value over the seeds, formatted as the seed-1
macro it accompanies.
"""
import json, os
import numpy as np
from analyse import merge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
A = json.load(open(os.path.join(RES, "audit.json")))
O = json.load(open(os.path.join(RES, "onboard.json")))
SPREAD = json.load(open(os.path.join(RES, "seed-spread.json")))
L = []

TAG = {"irrigation_timing": "Irr", "fertilizer_timing": "Fert", "pest_risk": "Pest"}
BEST_LATENT = {"irrigation_timing": "Dr", "fertilizer_timing": "gdd",
               "pest_risk": "sv_acc"}


def cmd(name, value):
    L.append(r"\newcommand{\%s}{%s}" % (name, value))


def sg(x, d=3):
    """Signed, except that a value rounding to zero carries no sign."""
    v = round(x, d)
    if abs(v) < 0.5 * 10 ** -d:
        return "$%.*f$" % (d, 0.0)
    return "$%+.*f$" % (d, v)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sci(x, d=1):
    if x <= 0:
        return "$0$"
    e = int(np.floor(np.log10(x)))
    return r"$%.*f\times 10^{%d}$" % (d, x / 10 ** e, e)


c = A["config"]
cmd("Years", c["years"]); cmd("PlantDoy", c["plant_doy"])
cmd("Folds", c["folds"]); cmd("Embargo", c["embargo"])
cmd("NDays", f"{c['n_days']:,}"); cmd("NFeatures", c["n_features"])
cmd("NSensorFeatures", c["n_features"] - 2)

flip_first = {}
for h, t in TAG.items():
    r = A["heads"][h]
    cmd(f"{t}Base", pct(r["base_rate"]))
    cmd(f"{t}Acc", pct(r["sensors"]["acc"][0]))
    cmd(f"{t}Maj", pct(r["sensors"]["maj"][0]))
    cmd(f"{t}Mcc", sg(r["sensors"]["mcc"][0]))
    cmd(f"{t}MccSd", f"{r['sensors']['mcc'][1]:.3f}")
    cmd(f"{t}Ceil", sg(r["sensors"]["ceil"][0]))
    cmd(f"{t}Cal", sg(r["calendar_only"]["mcc"][0]))
    cmd(f"{t}NoCal", sg(r["no_calendar"]["mcc"][0]))
    cmd(f"{t}Oracle", sg(r["oracle"][BEST_LATENT[h]]["mcc"][0]))
    cmd(f"{t}Rain", sg(r["addable"]["rain"]["mcc"][0]))
    cmd(f"{t}RainThirty", sg(r["addable"]["rain_30d"]["mcc"][0]))
    cmd(f"{t}RainNinety", sg(r["addable"]["rain_90d"]["mcc"][0]))
    cmd(f"{t}RainWindows", sg(r["addable"]["rain_windows"]["mcc"][0]))
    cmd(f"{t}PminusEto", sg(r["addable"]["p_minus_eto"]["mcc"][0]))
    cmd(f"{t}Brier", f"{r['sensors']['brier'][0]:.4f}")
    q = r["quant"]
    i15 = q["bits"].index(15)
    cmd(f"{t}FlipMeas", sci(q["meas"][i15]))
    cmd(f"{t}FlipBound", sci(q["bnd"][i15]))
    # the largest word length at which any decision changes: below Q15, this is
    # where quantisation first becomes visible as the word length is reduced
    first = max((b for b, v in zip(q["bits"], q["meas"]) if v > 0), default=None)
    cmd(f"{t}FlipFirstBits", first if first is not None else "none")
    flip_first[t] = first

cmd("FlipFirstBitsMax", max(b for b in flip_first.values() if b is not None))
cmd("FlipMarginBits", 15 - max(b for b in flip_first.values() if b is not None))

# the on-board recursion
v = O["variants"]
cmd("OnFullMcc", sg(v["full"]["mcc"]))
cmd("OnCalMcc", sg(v["hargreaves calibrated"]["mcc"]))
cmd("OnRawMcc", sg(v["hargreaves"]["mcc"]))
cmd("OnNoRainMcc", sg(v["calibrated, no rain gauge"]["mcc"]))
cmd("OnBestMcc", sg(O["bias_sweep_best_mcc"]))
cmd("EtoBias", f"{O['eto_bias_mm_per_day']:.2f}")
cmd("EtoBiasPct", f"{100 * O['eto_bias_mm_per_day'] / O['eto_mean_mm_per_day']:.0f}")
cmd("EtoRmse", f"{O['eto_rmse_mm_per_day']:.2f}")
cmd("EtoBiasCal", f"{O['eto_bias_cal_mm_per_day']:+.3f}")
cmd("SeasonDays", O["season_days"])
cmd("SeasonBias", f"{O['eto_bias_mm_per_day'] * O['season_days']:.0f}")
cmd("RawMm", f"{O['raw_mm']:.0f}"); cmd("TawMm", f"{O['taw_mm']:.0f}")

e = O["events"]
cmd("EvTrue", e["n_true"]); cmd("EvPred", e["n_pred"])
cmd("EvPerSeasonTrue", f"{e['n_true'] / e['seasons']:.1f}")
cmd("EvPerSeasonPred", f"{e['n_pred'] / e['seasons']:.1f}")
cmd("EvMedianOffset", f"{e['median_offset_days']:.0f}")
cmd("EvWithinThree", pct(e["within_3_days"], 0))
# LaTeX control sequences cannot contain digits, so the tolerance is spelled out
WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 5: "Five", 7: "Seven",
        10: "Ten", 14: "Fourteen"}
for t in e["tol"]:
    w = WORD[t["tol"]]
    cmd(f"EvFone{w}", f"{t['f1']:.3f}")
    cmd(f"EvRecall{w}", f"{t['recall']:.3f}")

# ---------------------------------------------------------------------------
# spread over weather-generator seeds
NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
           7: "seven", 8: "eight", 9: "nine", 10: "ten"}
IRR, FERT, PEST = "irrigation_timing", "fertilizer_timing", "pest_risk"


def first_flip(a, h):
    q = a["heads"][h]["quant"]
    return max((b for b, v in zip(q["bits"], q["meas"]) if v > 0), default=0)


def q15(a, h, key):
    q = a["heads"][h]["quant"]
    return q[key][q["bits"].index(15)]


mcc = lambda a, h, key: a["heads"][h][key]["mcc"][0]
# name: (value from one seed's merged audit record and onboard record, formatter)
SPREAD_ITEMS = {
    "IrrMcc": (lambda a, o: mcc(a, IRR, "sensors"), sg),
    "IrrCeil": (lambda a, o: a["heads"][IRR]["sensors"]["ceil"][0], sg),
    "IrrOracle": (lambda a, o: a["heads"][IRR]["oracle"]["Dr"]["mcc"][0], sg),
    "IrrAddBest": (lambda a, o: max(v["mcc"][0] for v in a["heads"][IRR]["addable"].values()), sg),
    "FertCal": (lambda a, o: mcc(a, FERT, "calendar_only"), sg),
    "FertNoCal": (lambda a, o: mcc(a, FERT, "no_calendar"), sg),
    "FertMcc": (lambda a, o: mcc(a, FERT, "sensors"), sg),
    "FertOracle": (lambda a, o: a["heads"][FERT]["oracle"]["gdd"]["mcc"][0], sg),
    "PestCal": (lambda a, o: mcc(a, PEST, "calendar_only"), sg),
    "PestNoCal": (lambda a, o: mcc(a, PEST, "no_calendar"), sg),
    "PestMcc": (lambda a, o: mcc(a, PEST, "sensors"), sg),
    "PestOracle": (lambda a, o: a["heads"][PEST]["oracle"]["sv_acc"]["mcc"][0], sg),
    "OnCalMcc": (lambda a, o: o["variants"]["hargreaves calibrated"]["mcc"], sg),
    "OnBestMcc": (lambda a, o: o["bias_sweep_best_mcc"], sg),
    "EvMedianOffset": (lambda a, o: o["events"]["median_offset_days"], lambda x: f"{x:.0f}"),
    "EvFoneZero": (lambda a, o: o["events"]["tol"][0]["f1"], lambda x: f"{x:.3f}"),
    "EvFoneThree": (lambda a, o: next(t["f1"] for t in o["events"]["tol"] if t["tol"] == 3),
                    lambda x: f"{x:.3f}"),
    "FlipMeasAll": (lambda a, o: max(q15(a, h, "meas") for h in TAG), sci),
    "FlipBoundAll": (lambda a, o: max(q15(a, h, "bnd") for h in TAG), sci),
    "FlipFirstBitsMax": (lambda a, o: max(first_flip(a, h) for h in TAG), lambda x: f"{x}"),
    "FlipMarginBits": (lambda a, o: 15 - max(first_flip(a, h) for h in TAG), lambda x: f"{x}"),
}

seeds = sorted(SPREAD, key=int)
per_seed = {}
for sd in seeds:
    r = SPREAD[sd]
    a = merge(r["trap.json"], r["decomposition.json"], r["quant.json"])
    per_seed[sd] = (a, r["onboard.json"])

# seed 1 in the spread must be the reported run itself
for name, (f, fmt) in SPREAD_ITEMS.items():
    assert fmt(f(*per_seed["1"])) == fmt(f(A, O)), f"seed 1 of the spread disagrees on {name}"

cmd("NSeeds", NUMWORD[len(seeds)])
for name, (f, fmt) in SPREAD_ITEMS.items():
    vals = [f(*per_seed[sd]) for sd in seeds]
    cmd(f"{name}Lo", fmt(min(vals)))
    cmd(f"{name}Hi", fmt(max(vals)))
# seed-1 values for the spread items that have no macro of their own above
defined = {l.split("{")[1][1:-1] for l in L}
for name, (f, fmt) in SPREAD_ITEMS.items():
    if name not in defined:
        cmd(name, fmt(f(A, O)))

# Is the calendar alone at least as good as the deployed feature set for
# fertilizer timing? Stated as a clause, so the sentence cannot claim more than
# the seeds show.
gap = [mcc(per_seed[sd][0], FERT, "calendar_only") - mcc(per_seed[sd][0], FERT, "sensors")
       for sd in seeds]
above = sum(g > 0 for g in gap)
if above == len(seeds):
    clause = "at every seed"
else:
    short = max(-g for g in gap if g <= 0)
    rest = len(seeds) - above
    clause = (f"at {NUMWORD[above]} of the {NUMWORD[len(seeds)]} seeds and falls short of it "
              + (f"by {short:.3f} at the other" if rest == 1
                 else f"by at most {short:.3f} at the other {NUMWORD[rest]}"))
cmd("FertCalAboveSeeds", clause)

out = os.path.join(RES, "numbers.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("% generated by code/build_numbers.py; do not edit\n")
    fh.write("\n".join(L) + "\n")
print(f"wrote {len(L)} macros")

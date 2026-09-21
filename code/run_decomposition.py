"""Where each decision's information actually is.

Each decision is scored four ways on the same folds: from the calendar alone, from
the sensed variables with the calendar removed, and with each latent state of the
reference policy supplied as an extra column. Then the inputs a rain gauge could
provide are added one at a time, from daily rainfall up to rainfall minus
evapotranspiration accumulated since planting.

    python3 code/run_decomposition.py    # writes results/decomposition.json
"""
import json, os
import numpy as np
from record import make_record, run_reference, YEARS, SEED, PLANT_DOY
from features import make_features
from protocol import score_head, HEADS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def rainfall_inputs(sim, dap):
    """What a rain gauge could give the classifier, as named columns.

    Daily rainfall, reference ETo beside it, trailing sums over 7, 30 and 90 days,
    all of those together, and rainfall minus Penman-Monteith ETo accumulated from
    planting and restarted each season. The last uses an ETo the node cannot
    compute, so it is the most generous accumulated input on offer.
    """
    rain = sim["PRECTOTCORR"]
    rsum = lambda L: np.convolve(rain, np.ones(L), mode="full")[:len(rain)]
    season_id = np.cumsum(np.asarray(dap) == 1)
    p_minus_eto = np.zeros(len(rain))
    for s in np.unique(season_id[season_id > 0]):
        m = season_id == s
        p_minus_eto[m] = np.cumsum(rain[m] - sim["ETO"][m])
    return {"rain": rain,
            "rain+eto": np.column_stack([rain, sim["ETO"]]),
            "rain_7d": rsum(7), "rain_30d": rsum(30), "rain_90d": rsum(90),
            "rain_windows": np.column_stack([rain, rsum(7), rsum(30), rsum(90)]),
            "p_minus_eto": p_minus_eto}


def main(seed=SEED, out=OUT):
    sim = make_record(YEARS, seed)
    lab, lat, grow = run_reference(sim, PLANT_DOY)
    X, names = make_features(sim)
    Xg = X[grow]
    doy_cols = [i for i, n in enumerate(names) if n.startswith("doy_")]
    sens_cols = [i for i in range(len(names)) if i not in doy_cols]
    latents = {"Dr": lat["Dr"], "gdd": lat["gdd"], "sv_acc": lat["sv_acc"]}
    addable = rainfall_inputs(sim, lat["dap"])
    S = {"heads": {}}
    for h in HEADS:
        y = np.asarray(lab[h])[grow].astype(int)
        rec = {"no_calendar": score_head(Xg[:, sens_cols], y),
               "calendar_only": score_head(Xg[:, doy_cols], y),
               "oracle": {}, "addable": {}}
        for nm, v in latents.items():
            rec["oracle"][nm] = score_head(np.column_stack([Xg, np.asarray(v)[grow]]), y)
        for nm, v in addable.items():
            v = np.asarray(v).reshape(len(v), -1)
            rec["addable"][nm] = score_head(np.column_stack([Xg, v[grow]]), y)
        S["heads"][h] = rec
        print(f"{h:<20} calendar {rec['calendar_only']['mcc'][0]:+.3f}  "
              f"no calendar {rec['no_calendar']['mcc'][0]:+.3f}  " + "  ".join(
              f"+{k} {rec['oracle'][k]['mcc'][0]:+.3f}" for k in latents))
        print(" " * 20 + " " + "  ".join(
              f"+{k} {rec['addable'][k]['mcc'][0]:+.3f}" for k in addable))
    os.makedirs(out, exist_ok=True)
    json.dump(S, open(os.path.join(out, "decomposition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

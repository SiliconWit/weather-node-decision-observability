"""Run the recursion on the node instead of learning it.

Variants, each stricter about what the node is allowed to know: the reference
inputs; Hargreaves ETo, uncalibrated and calibrated on the first five years;
Penman-Monteith with radiation estimated from the temperature range, with the
FAO-56 interior coefficient and with the coefficient calibrated on the same five
years; Hargreaves rescaled to the mean of the uncalibrated Penman-Monteith
estimate, which separates mean bias from day-to-day error; the calibrated
Hargreaves variant with a residual bias, and with no rain gauge. Then a sweep of residual bias, and each node-side ETo estimate scored as
events at several tolerances.

    python3 code/run_onboard.py      # writes results/onboard.json
"""
import json, os
import numpy as np
from sklearn.metrics import matthews_corrcoef
from record import make_record, run_reference, LAT, YEARS, SEED, PLANT_DOY
from onboard import (eto_hargreaves, eto_pm_node, calibrate_krs, onboard_decision,
                     event_match, KRS_INTERIOR)
import reference as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def score_events(pred, y, tols=(0, 1, 2, 3, 5, 7, 10, 14)):
    """Event counts, signed offsets and tolerance-scored agreement for one schedule.

    The offset is from each reference event to the nearest predicted one, so that
    late and early are distinguishable rather than both counted as wrong.
    """
    ev = dict(n_true=int(y.sum()), n_pred=int(pred.sum()), seasons=int(round(YEARS)), tol=[])
    pi, ti = np.flatnonzero(pred), np.flatnonzero(y)
    offs = [int(pi[np.argmin(np.abs(pi - t))] - t) for t in ti] if len(pi) and len(ti) else []
    ev["offsets"] = offs
    ev["median_offset_days"] = float(np.median(np.abs(offs))) if offs else None
    ev["within_3_days"] = float(np.mean(np.abs(offs) <= 3)) if offs else None
    for tol in tols:
        e = event_match(pred, y, tol)
        e["tol"] = tol
        ev["tol"].append(e)
    return ev


def main(seed=SEED, out=OUT):
    sim = make_record(YEARS, seed=seed)
    lab, lat, grow = run_reference(sim, plant_doy=PLANT_DOY)
    y = np.asarray(lab["irrigation_timing"])[grow].astype(int)
    # the reference runs the balance continuously, clamping dap to at least 1
    dap = np.maximum(lat["dap"], 1)
    rain, doy = sim["PRECTOTCORR"], sim["doy"]

    eto_h = eto_hargreaves(sim["T2M_MAX"], sim["T2M_MIN"], doy, LAT)
    # Calibrate the Hargreaves coefficient on the first five years only, which is
    # what a deployment could do against a reference station before shipping.
    cal = len(doy) // 8
    k = float(sim["ETO"][:cal].mean() / eto_h[:cal].mean())
    eto_c = k * eto_h
    # Penman-Monteith from the node's own temperature, humidity, wind and pressure,
    # with radiation estimated from the temperature range (FAO-56 eq. 50)
    node = (sim["T2M_MAX"], sim["T2M_MIN"], sim["RH2M"], sim["WS2M"], sim["PS"])
    eto_p = eto_pm_node(*node, doy, LAT, KRS_INTERIOR)
    krs = calibrate_krs(sim["ETO"][:cal], *(v[:cal] for v in node), doy[:cal], LAT)
    eto_pc = eto_pm_node(*node, doy, LAT, krs)
    # Hargreaves rescaled to the uncalibrated Penman-Monteith mean over the same
    # five years: equal mean bias, Hargreaves day-to-day error. Needs no reference.
    k_pm = float(eto_p[:cal].mean() / eto_h[:cal].mean())
    eto_hp = k_pm * eto_h
    estimates = {"hargreaves": eto_h, "hargreaves calibrated": eto_c,
                 "penman-monteith, estimated Rs": eto_p,
                 "penman-monteith, calibrated k_Rs": eto_pc,
                 "hargreaves, rescaled to penman-monteith mean": eto_hp}
    variants = {
        "full": (sim["ETO"], rain),
        "hargreaves": (eto_h, rain),
        "hargreaves calibrated": (eto_c, rain),
        "penman-monteith, estimated Rs": (eto_p, rain),
        "penman-monteith, calibrated k_Rs": (eto_pc, rain),
        "hargreaves, rescaled to penman-monteith mean": (eto_hp, rain),
        "calibrated, 2% high": (1.02 * eto_c, rain),
        "calibrated, 5% high": (1.05 * eto_c, rain),
        "calibrated, no rain gauge": (eto_c, np.zeros_like(rain)),
    }
    season = sum(R.L.values())
    S = {"eto_mean_mm_per_day": float(np.mean(sim["ETO"])),
         "eto_bias_mm_per_day": float(np.mean(eto_h - sim["ETO"])),
         "eto_rmse_mm_per_day": float(np.sqrt(np.mean((eto_h - sim["ETO"]) ** 2))),
         "eto_cal_factor": k,
         "eto_bias_cal_mm_per_day": float(np.mean(eto_c - sim["ETO"])),
         "krs": KRS_INTERIOR, "krs_cal": krs,
         "season_days": int(season),
         "eto_estimates": {},
         "variants": {}}
    print(f"{'node ETo estimate':<46}{'bias mm/d':>10}{'RMSE mm/d':>11}")
    for nm, e in estimates.items():
        b = float(np.mean(e - sim["ETO"]))
        r = float(np.sqrt(np.mean((e - sim["ETO"]) ** 2)))
        S["eto_estimates"][nm] = dict(bias_mm_per_day=b, rmse_mm_per_day=r)
        print(f"{nm:<46}{b:>+10.3f}{r:>11.3f}")
    print(f"k_Rs calibrated on the first five years: {krs:.4f} "
          f"(FAO-56 interior value {KRS_INTERIOR})\n")
    # what the threshold is, so that an accumulated bias can be read against it
    taw = 1000.0 * (R.THETA_FC - R.THETA_WP) * R.root_depth(season // 2)
    raw = float(np.clip(R.P_TABLE + 0.04 * (5.0 - 3.5), 0.1, 0.8)) * taw
    S["taw_mm"], S["raw_mm"] = float(taw), float(raw)
    print(f"mid-season TAW {taw:.0f} mm, RAW (the threshold) {raw:.0f} mm; "
          f"a season is {season} days")
    print(f"uncalibrated bias accumulates to "
          f"{S['eto_bias_mm_per_day'] * season:.0f} mm over one season\n")
    print(f"{'node runs the recursion with':<46}{'MCC':>8}{'accuracy':>10}"
          f"{'recall':>9}{'precision':>11}")
    for nm, (e, rr) in variants.items():
        d = onboard_decision(e, rr, dap)[grow]
        tp = int(((d == 1) & (y == 1)).sum())
        rec = tp / max(int((y == 1).sum()), 1)
        pre = tp / max(int((d == 1).sum()), 1)
        m = float(matthews_corrcoef(y, d))
        S["variants"][nm] = dict(mcc=m, acc=float((d == y).mean()),
                                 recall=float(rec), precision=float(pre),
                                 n_pos_pred=int((d == 1).sum()),
                                 n_pos_true=int((y == 1).sum()))
        print(f"{nm:<46}{m:>+8.3f}{(d == y).mean():>10.3f}{rec:>9.3f}{pre:>11.3f}")

    # How much residual bias can the recursion tolerate? The decision variable is
    # an integral, so this is the curve that sets the sensor specification.
    print("\nresidual ETo bias against decision agreement")
    print(f"{'multiplier':>11}{'bias mm/d':>11}{'season mm':>11}{'MCC':>8}")
    sweep = []
    for mult in (0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10):
        e = mult * eto_c
        bias = float(np.mean(e - sim["ETO"]))
        d = onboard_decision(e, rain, dap)[grow]
        m = float(matthews_corrcoef(y, d))
        sweep.append(dict(mult=mult, bias=bias, season_mm=bias * season, mcc=m))
        print(f"{mult:>11.2f}{bias:>11.3f}{bias * season:>11.1f}{m:>+8.3f}")
    S["bias_sweep"] = sweep
    S["bias_sweep_best_mcc"] = float(max(x["mcc"] for x in sweep))

    # Is the node wrong, or a couple of days late? This is the question a per-day
    # score cannot answer, and the one an irrigation manager actually asks.
    pred = onboard_decision(eto_c, rain, dap)[grow]
    S["events"] = score_events(pred, y)
    print("\nagreement when a prediction within a tolerance counts as a hit "
          "(Hargreaves calibrated)")
    print(f"{'tolerance':>10}{'recall':>9}{'precision':>11}{'F1':>8}")
    for e in S["events"]["tol"]:
        print(f"{e['tol']:>7} d{e['recall']:>9.3f}{e['precision']:>11.3f}{e['f1']:>8.3f}")
    print(f"\nreference calls {y.sum() / round(YEARS):.1f} irrigations per season, "
          f"the node calls {pred.sum() / round(YEARS):.1f}; median offset "
          f"{S['events']['median_offset_days']:.0f} days")

    # the same event scoring for every node-side ETo estimate
    S["events_by_estimate"] = {}
    print(f"\n{'events, by node ETo estimate':<46}{'per season':>11}{'median off':>11}"
          f"{'F1 0 d':>8}{'F1 3 d':>8}{'F1 7 d':>8}")
    for nm, e in estimates.items():
        ev = score_events(onboard_decision(e, rain, dap)[grow], y)
        S["events_by_estimate"][nm] = ev
        f1 = {t["tol"]: t["f1"] for t in ev["tol"]}
        print(f"{nm:<46}{ev['n_pred'] / ev['seasons']:>11.1f}"
              f"{ev['median_offset_days']:>11.0f}{f1[0]:>8.3f}{f1[3]:>8.3f}{f1[7]:>8.3f}")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "onboard.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    print(f"\nwrote {os.path.join(out, 'onboard.json')}")


if __name__ == "__main__":
    main()

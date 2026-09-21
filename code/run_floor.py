"""Does the simulated record look like the place it was fitted to?

Compares the generator's output with the observed record on the statistics the
decisions depend on, then runs the reference policy and reports how often each
decision is called.

    python3 code/run_floor.py        # writes results/floor.json
"""
import json, os
import numpy as np
from record import load_observed, make_record, run_reference, YEARS, SEED, PLANT_DOY
import reference as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def climate_stats(doy, tmax, tmin, rh, rain):
    """Annual rainfall, wet-day fraction, mean temperatures and humidity, and the
    day-to-day persistence of maximum temperature.

    A wet day is one with at least 1 mm. Persistence is the lag-one correlation of
    the maximum-temperature series; a generator that gets the means right and the
    persistence wrong will produce the right climate with the wrong weather.
    """
    n_years = len(doy) / 365.25
    return dict(rain_mm_per_year=float(rain.sum() / n_years),
                wet_day_fraction=float((rain >= 1.0).mean()),
                tmax_mean=float(tmax.mean()), tmin_mean=float(tmin.mean()),
                rh_mean=float(rh.mean()),
                tmax_lag1=float(np.corrcoef(tmax[:-1], tmax[1:])[0, 1]))


def main(seed=SEED, out=OUT):
    doy_o, data_o, rain_o = load_observed()
    sim = make_record(YEARS, seed)
    obs = climate_stats(doy_o, data_o["T2M_MAX"], data_o["T2M_MIN"], data_o["RH2M"], rain_o)
    gen = climate_stats(sim["doy"], sim["T2M_MAX"], sim["T2M_MIN"], sim["RH2M"],
                        sim["PRECTOTCORR"])
    gen["eto_mean_mm_per_day"] = float(sim["ETO"].mean())
    lab, lat, grow = run_reference(sim, PLANT_DOY)
    seasons = int((lat["dap"] == 1).sum())
    ref = dict(n_growing_days=int(grow.sum()), seasons=seasons,
               base_rate={h: float(np.asarray(lab[h])[grow].mean())
                          for h in ("irrigation_timing", "fertilizer_timing", "pest_risk")},
               irrigation_days_per_season=float(lab["irrigation_timing"][grow].sum() / seasons),
               season_days=int(sum(R.L.values())))
    S = dict(config=dict(years=YEARS, seed=seed, plant_doy=PLANT_DOY),
             observed=obs, simulated=gen, reference=ref)
    os.makedirs(out, exist_ok=True)
    json.dump(S, open(os.path.join(out, "floor.json"), "w"), indent=1)
    for k in obs:
        print(f"  {k:<20}{obs[k]:>10.3f}{gen[k]:>10.3f}")
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()

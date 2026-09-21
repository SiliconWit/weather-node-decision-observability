"""The quantization error budget.

For the deployed feature set, on the same folds, the measured, estimated and
worst-case bounded decision flip rate at every word length from 3 to 16
fractional bits, and the distribution of margins that the bound depends on.

    python3 code/run_quant.py        # writes results/quant.json
"""
import json, os
import numpy as np
from record import make_record, run_reference, YEARS, SEED, PLANT_DOY
from features import make_features
from protocol import blocked_splits, scaled, fit_lr, HEADS
import quantize as Q

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")
BITS = list(range(3, 17))


def main(seed=SEED, out=OUT):
    sim = make_record(YEARS, seed)
    lab, lat, grow = run_reference(sim, PLANT_DOY)
    X, names = make_features(sim)
    Xg = X[grow]
    S = {"bits": BITS, "heads": {}}
    for h in HEADS:
        y = np.asarray(lab[h])[grow].astype(int)
        flips = {k: [] for k in ("meas", "est", "bnd")}
        margins = []
        for tr, te in blocked_splits(len(y)):
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr, Xte = scaled(Xg[tr], Xg[te])
            w, b = fit_lr(Xtr, y[tr])
            margins.append(np.abs(Xte @ w + b))
            for fb in BITS:
                flips["meas"].append((fb, Q.flip_rate_measured(Xte, w, b, fb)))
                flips["est"].append((fb, Q.flip_rate_estimate(Xte, w, b, fb)))
                flips["bnd"].append((fb, Q.flip_rate_bound(Xte, w, b, fb)))
        q = {"bits": BITS}
        for k, pairs in flips.items():
            q[k] = [float(np.mean([v for b_, v in pairs if b_ == fb])) for fb in BITS]
        m = np.concatenate(margins)
        S["heads"][h] = {"quant": q,
                         "margin": dict(n=int(m.size), median=float(np.median(m)),
                                        p01=float(np.percentile(m, 1)),
                                        p10=float(np.percentile(m, 10)))}
        print(f"{h:<20} measured at Q15 {q['meas'][BITS.index(15)]:.2g}  "
              f"bound at Q15 {q['bnd'][BITS.index(15)]:.2g}")
    os.makedirs(out, exist_ok=True)
    json.dump(S, open(os.path.join(out, "quant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Score the deployed model by accuracy and by Matthews correlation.

For each decision, accuracy, the rate achieved by always predicting the majority
class on the same split, the Matthews correlation, and a high-capacity reference
fitted on identical features. Read the accuracy only beside the majority rate.

    python3 code/run_trap.py         # writes results/trap.json
"""
import json, os
import numpy as np
from record import make_record, run_reference, YEARS, SEED, PLANT_DOY
from features import make_features
from protocol import score_head, HEADS, FOLDS, EMBARGO, C_REG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def main(seed=SEED, out=OUT):
    sim = make_record(YEARS, seed)
    lab, lat, grow = run_reference(sim, PLANT_DOY)
    X, names = make_features(sim)
    Xg = X[grow]
    S = {"config": dict(years=YEARS, seed=seed, plant_doy=PLANT_DOY, folds=FOLDS,
                        embargo=EMBARGO, C=C_REG, n_days=int(Xg.shape[0]),
                        n_features=int(Xg.shape[1]), feature_names=names),
         "heads": {}}
    for h in HEADS:
        y = np.asarray(lab[h])[grow].astype(int)
        S["heads"][h] = {"base_rate": float(y.mean()),
                         "sensors": score_head(Xg, y, ceiling=True)}
        r = S["heads"][h]["sensors"]
        print(f"{h:<20} acc {r['acc'][0]:.3f}  majority {r['maj'][0]:.3f}  "
              f"MCC {r['mcc'][0]:+.3f}  capacity reference {r['ceil'][0]:+.3f}")
    os.makedirs(out, exist_ok=True)
    json.dump(S, open(os.path.join(out, "trap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

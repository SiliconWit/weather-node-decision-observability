"""Repeat every experiment at several weather-generator seeds.

Seed 1 is the reported run. The generator is stochastic, so each number also has
a spread over seeds. This
script reruns the floor, trap, decomposition, quantization and on-board
experiments at each seed and collects them in one file.

    python3 code/run_seeds.py              # writes results/seed-spread.json
    python3 code/run_seeds.py --workers 2  # fewer parallel processes

Each seed is an independent run of the full pipeline, so the seeds are spread over
worker processes. Per-seed results are kept under results/seeds/.
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

import run_floor, run_trap, run_decomposition, run_quant, run_onboard

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
SEEDS = [1, 2, 3, 4, 5, 6]
FILES = ("floor.json", "trap.json", "decomposition.json", "quant.json", "onboard.json")


def one(seed):
    out = os.path.join(RES, "seeds", f"seed{seed}")
    for mod in (run_floor, run_trap, run_decomposition, run_quant, run_onboard):
        mod.main(seed=seed, out=out)
    return {n: json.load(open(os.path.join(out, n))) for n in FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(zip((str(s) for s in SEEDS), ex.map(one, SEEDS)))
    json.dump(res, open(os.path.join(RES, "seed-spread.json"), "w"))
    print(f"wrote {os.path.join(RES, 'seed-spread.json')}: seeds {SEEDS}")


if __name__ == "__main__":
    main()

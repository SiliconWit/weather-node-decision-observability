# Reckoner: what a low-cost weather node can decide

What a low-cost edge AI weather node can and cannot decide about irrigation, fertilizer
and spraying, separating what its sensors can observe from what fixed-point inference
loses.

## Requirements

Python 3.10 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

## Data

NASA POWER daily weather for Nyeri, Kenya, 2005 to 2024. The record used for every
result is committed in `data/`, with checksums and details in `data/MANIFEST.md`.
`data/fetch_power.py` downloads a fresh copy into `data/download/` for comparison.

## Running

From the repository root:

```bash
python3 code/run_floor.py
python3 code/run_trap.py
python3 code/run_decomposition.py
python3 code/run_quant.py
python3 code/analyse.py
python3 code/run_onboard.py
python3 code/run_seeds.py          # repeats the analysis at six weather-generator seeds
python3 code/build_numbers.py
python3 code/figures.py
```

The full sequence takes about fifteen minutes on a multi-core CPU.

## Outputs

Everything is written to `results/`: the JSON results files, the per-seed runs in
`results/seeds/`, `numbers.tex` (every reported value as a LaTeX macro) and the figures
in `results/figs/`.

## Citation

See `CITATION.cff`.

## License

MIT

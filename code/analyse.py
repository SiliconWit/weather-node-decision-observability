"""Gather the per-experiment results files into results/audit.json.

    python3 code/analyse.py          # reads trap, decomposition, quant; writes results/audit.json

Nothing is recomputed here. If a number in audit.json is wrong, the fix belongs in
the script that produced it, not in this one.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def merge(trap, decomposition, quant):
    """One record per decision head, with the configuration alongside.

    Every head must appear in all three inputs; a head missing from one of them
    means a script was run on a different configuration, and that is an error
    rather than something to paper over.
    """
    out = {"config": trap["config"], "heads": {}}
    for h, t in trap["heads"].items():
        rec = dict(t)
        rec.update(decomposition["heads"][h])
        rec.update(quant["heads"][h])
        out["heads"][h] = rec
    return out


def main(res=RES):
    load = lambda n: json.load(open(os.path.join(res, n)))
    audit = merge(load("trap.json"), load("decomposition.json"), load("quant.json"))
    json.dump(audit, open(os.path.join(res, "audit.json"), "w"), indent=1)
    print(f"wrote {os.path.join(res, 'audit.json')}: {len(audit['heads'])} heads")


if __name__ == "__main__":
    main()

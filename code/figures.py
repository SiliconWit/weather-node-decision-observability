"""The figures, drawn from the results files. No number is retyped.

    python3 code/figures.py          # reads results/audit.json, onboard.json; writes results/figs/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figs")

NAVY, GOLD, SLATE, RED, MOSS = "#0F284D", "#B0892C", "#5A6473", "#8C2F1F", "#3E6B4F"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 200,
    "savefig.bbox": "tight"})

HEADS = ["irrigation_timing", "fertilizer_timing", "pest_risk"]
NICE = {"irrigation_timing": "irrigation\ntiming",
        "fertilizer_timing": "fertilizer\ntiming", "pest_risk": "pest\nrisk"}
BEST_LATENT = {"irrigation_timing": "Dr", "fertilizer_timing": "gdd",
               "pest_risk": "sv_acc"}
LATENT_NICE = {"Dr": "depletion", "gdd": "thermal time", "sv_acc": "blight severity"}


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print(name)


def fig_observability(A):
    """Each decision scored from the calendar, the sensors without it, the sensors with it, and the latent state."""
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    x = np.arange(len(HEADS))
    w = 0.2
    series = [("calendar alone", "calendar_only", SLATE),
              ("sensors, calendar removed", "no_calendar", MOSS),
              ("sensors as deployed", "sensors", NAVY),
              ("with the latent state", None, GOLD)]
    for j, (lab, key, col) in enumerate(series):
        v = [A["heads"][h]["oracle"][BEST_LATENT[h]]["mcc"][0] if key is None
             else A["heads"][h][key]["mcc"][0] for h in HEADS]
        ax.bar(x + (j - 1.5) * w, v, w, color=col, label=lab, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[h] for h in HEADS])
    ax.set_ylabel("Matthews correlation")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2,
              columnspacing=1.0)
    for i, h in enumerate(HEADS):
        ax.text(i + 1.5 * w, A["heads"][h]["oracle"][BEST_LATENT[h]]["mcc"][0] + 0.03,
                LATENT_NICE[BEST_LATENT[h]], ha="center", fontsize=6.2, color=GOLD)
    _save(fig, "fig1_observability.pdf")


def fig_accuracy_trap(A):
    """Accuracy beside the majority-class rate, and Matthews correlation beside it."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    x = np.arange(len(HEADS))
    acc = [A["heads"][h]["sensors"]["acc"][0] for h in HEADS]
    maj = [A["heads"][h]["sensors"]["maj"][0] for h in HEADS]
    ax[0].bar(x - 0.2, acc, 0.4, color=NAVY, label="the model", edgecolor="none")
    ax[0].bar(x + 0.2, maj, 0.4, color=SLATE, label="always the majority class",
              edgecolor="none")
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 1.28)
    ax[0].legend(frameon=False, loc="upper center", ncol=1, handlelength=1.2)
    mcc = [A["heads"][h]["sensors"]["mcc"][0] for h in HEADS]
    sd = [A["heads"][h]["sensors"]["mcc"][1] for h in HEADS]
    ax[1].bar(x, mcc, 0.45, yerr=sd, capsize=2.5, color=NAVY, edgecolor="none",
              error_kw=dict(lw=0.8))
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("Matthews correlation"); ax[1].set_ylim(-0.1, 1.0)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([NICE[h] for h in HEADS])
    ax[0].set_title("what accuracy reports")
    ax[1].set_title("what skill reports")
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig2_accuracy_trap.pdf")


def fig_timing(O):
    """Event agreement against tolerance, and the distribution of signed timing offsets."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    tol = [e["tol"] for e in O["events"]["tol"]]
    f1 = [e["f1"] for e in O["events"]["tol"]]
    rec = [e["recall"] for e in O["events"]["tol"]]
    ax[0].plot(tol, f1, marker="o", ms=3.5, color=NAVY, label="F1")
    ax[0].plot(tol, rec, marker="s", ms=3.5, color=GOLD, ls="--", label="recall")
    ax[0].scatter([0], [f1[0]], s=42, facecolor="white", edgecolor=RED, zorder=5,
                  lw=1.2)
    ax[0].annotate("what a per-day\nscore reports", (0, f1[0]), (2.1, 0.16),
                   fontsize=6.6, color=RED,
                   arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax[0].set_xlabel("tolerance (days)"); ax[0].set_ylabel("agreement")
    ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False, loc="lower right")
    ax[0].set_title("scheduling the same irrigations")

    off = np.array(O["events"]["offsets"])
    bins = np.arange(-14.5, 15.5, 1.0)
    ax[1].hist(off, bins=bins, color=NAVY, edgecolor="none")
    ax[1].axvline(0, color=RED, lw=1.0, ls=":")
    ax[1].set_xlabel("days, reference event to nearest prediction")
    ax[1].set_ylabel("events")
    ax[1].set_xlim(-14, 14)
    ax[1].set_title("and how late it is")
    w3 = O["events"]["within_3_days"]
    ax[1].text(0.97, 0.92, f"{100 * w3:.0f}% within 3 days", transform=ax[1].transAxes,
               ha="right", fontsize=6.8, color=SLATE)
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig3_timing.pdf")


def fig_quant(A):
    """Measured, estimated and bounded flip rate against word length, per head."""
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=True)
    for a, h in zip(ax, HEADS):
        q = A["heads"][h]["quant"]
        floor = 1.0 / A["heads"][h]["margin"]["n"]
        for key, lab, col, mk, ls in (
                ("bnd", "worst-case bound", RED, "^", "-"),
                ("est", "uniform-error estimate", GOLD, "s", "--"),
                ("meas", "measured", NAVY, "o", "-")):
            a.semilogy(q["bits"], np.maximum(q[key], floor / 10), color=col,
                       marker=mk, ms=2.6, ls=ls, label=lab)
        a.axhline(floor, color=SLATE, lw=0.8, ls=":", label="one sample in the test set")
        a.axvline(15, color=MOSS, lw=0.9, ls="-.")
        a.text(15.3, 0.25, "Q15", color=MOSS, fontsize=6.4, rotation=90, va="center")
        a.set_xlabel("fractional bits")
        a.set_title(NICE[h].replace("\n", " "), fontsize=8)
    ax[0].set_ylabel("decision flip rate")
    hs, ls_ = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls_, loc="upper center", ncol=4, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.9))
    _save(fig, "fig4_quantization.pdf")


def main():
    A = json.load(open(os.path.join(RES, "audit.json")))
    O = json.load(open(os.path.join(RES, "onboard.json")))
    fig_observability(A)
    fig_accuracy_trap(A)
    fig_timing(O)
    fig_quant(A)


if __name__ == "__main__":
    main()

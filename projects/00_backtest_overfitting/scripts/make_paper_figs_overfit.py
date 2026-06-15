#!/usr/bin/env python3
"""Reproduce the two paper figures that summarize the overfitting corpus, directly from
the committed table ``tables/cross_market_summary.csv``:

  fig_luck.pdf  — best OOS Sharpe vs the skill-less expected maximum (luck bar), per market.
  fig_effn.pdf  — nominal strategy count vs effective independent bets, per market.

Both read only saved artifacts, so every number on them is reproducible. Usage:
    python3 make_paper_figs_overfit.py [out_dir]
"""
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CSV = os.path.join(PROJ, "tables", "cross_market_summary.csv")

BLUE = "#2c7fb8"; GREY = "#9a9a9a"; RED = "#c0392b"
LABELS = {"crypto": "Crypto", "equity": "US Equity", "fx": "FX"}

_BASE_RC = {
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
}


def load():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    order = {"crypto": 0, "equity": 1, "fx": 2}
    rows.sort(key=lambda r: order.get(r["market"], 9))
    return rows


def fig_luck(rows, out):
    plt.rcParams.update(_BASE_RC)
    import numpy as np
    x = np.arange(len(rows)); w = 0.38
    best = [float(r["best_SR_ann"]) for r in rows]
    emax = [float(r["null_E[max]_ann"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    b1 = ax.bar(x - w / 2, best, w, color=BLUE, label="Best observed (OOS)")
    b2 = ax.bar(x + w / 2, emax, w, color=GREY, label=r"Skill-less expected max $E[\max]$")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r["market"]] for r in rows])
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("The best strategy in every market is below the luck bar")
    ax.set_ylim(0, max(emax) * 1.15); ax.legend(loc="upper left")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_effn(rows, out):
    plt.rcParams.update(_BASE_RC)
    import numpy as np
    x = np.arange(len(rows)); w = 0.38
    nom = [int(r["n_strat"]) for r in rows]
    eff = [int(r["eff_N_pooled"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.bar(x - w / 2, nom, w, color=GREY, label="Nominal strategies")
    ax.bar(x + w / 2, eff, w, color=RED, label="Effective independent bets")
    ax.set_yscale("log"); ax.set_ylim(1, max(nom) * 8.0)
    for i, (n, e) in enumerate(zip(nom, eff)):
        ax.annotate(f"{n:,}", (i - w / 2, n), ha="center", va="bottom", fontsize=10, xytext=(0, 1), textcoords="offset points")
        ax.annotate(f"{e:,}", (i + w / 2, e), ha="center", va="bottom", fontsize=10, xytext=(0, 1), textcoords="offset points")
        ax.annotate(f"÷{round(n / e)}", (i + w / 2, e), ha="center", va="top", color=RED, fontsize=9,
                    xytext=(0, -3), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r["market"]] for r in rows])
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Apparent diversity is an illusion: the bet count collapses")
    ax.legend(loc="upper center", ncol=2)
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "figures")
    os.makedirs(out_dir, exist_ok=True)
    rows = load()
    fig_luck(rows, os.path.join(out_dir, "fig_luck.pdf"))
    fig_effn(rows, os.path.join(out_dir, "fig_effn.pdf"))

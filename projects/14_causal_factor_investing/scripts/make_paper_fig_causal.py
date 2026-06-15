#!/usr/bin/env python3
"""Regenerate the paper figure ``fig_causal.pdf`` (the fork dose-response panel)
from the committed Monte-Carlo sweep, with correct percent-sign rendering.

The earlier paper build passed LaTeX-escaped ``\\%`` into a non-usetex matplotlib
string, so the title and y-axis label printed a literal backslash. This script
reads the saved ``mc_sweep.csv`` and renders the single fork panel the paper
uses, with plain ``%`` in text and ``\\%`` only inside math mode.

Usage:
    python make_paper_fig_causal.py [output.pdf]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SWEEP = os.path.join(PROJ, "tables", "mc_sweep.csv")

# Colour-blind-safe palette (mirrors lib/style.py so the figure matches the rest).
ACCENT = "#D55E00"   # naive regression
GREEN = "#009E73"    # backdoor-adjusted
GREY = "#999999"     # nominal control

# Paper figure shows the confounder dose-response over [0, 0.6].
XMAX = 0.6


def main(out_path: str) -> None:
    df = pd.read_csv(SWEEP)
    df = df[df["strength"] <= XMAX + 1e-9].copy()
    x = df["strength"].to_numpy()
    naive = 100.0 * df["fork_naive_falsepos"].to_numpy()
    adj = 100.0 * df["fork_adj_falsepos"].to_numpy()

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(x, naive, "-o", color=ACCENT, lw=2.2, ms=7, label="Naive regression")
    ax.plot(x, adj, "-s", color=GREEN, lw=2.2, ms=6, label="Backdoor-adjusted")
    ax.axhline(5.0, color=GREY, ls="--", lw=1.4, label=r"Nominal $\alpha = 5\%$")

    ax.set_title("A hidden confounder fakes significance 100% of the time")
    ax.set_xlabel("Confounder strength")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_xlim(-0.02, XMAX + 0.02)
    ax.set_ylim(-3, 105)
    ax.legend(loc="center right")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "figures", "fig_causal.pdf")
    main(out)

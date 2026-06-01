#!/usr/bin/env python3
"""
Cross-market comparison for the 50k x 3 backtest-overfitting
study. Reads the three per-market corpus summaries (crypto / equity / fx) that
the at-scale harness already produced and emits:

  tables/cross_market_summary.md   (+ .csv)  -- one row per market
  figures/fig7_cross_market.png               -- best-vs-null (left) and
                                                  nominal vs effective-N (right)

Read-only on lib/*; works only inside projects/00_backtest_overfitting/.
No re-run of the heavy harness: the per-market CSVs are the source of truth.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.exists(_os.path.join(_d, "config.py")):
    _d = _os.path.dirname(_d)
REPO_ROOT = _d
_sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
_sys.path.insert(0, _LIB)
import style as ST
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKETS = ["crypto", "equity", "fx"]
LABEL = {"crypto": "Crypto", "equity": "US Equity", "fx": "Forex"}


def load():
    rows = []
    for m in MARKETS:
        s = pd.read_csv(f"{PROJ}/tables/corpus_summary_{m}.csv").iloc[0].to_dict()
        rows.append(s)
    return pd.DataFrame(rows)


def main():
    df = load()
    # cross-market table (raw numbers straight from the per-market summaries)
    cols = ["market", "n_strategies", "n_pairs", "best_sr_ann", "sr0_ann",
            "dsr", "median_pbo", "eff_n_pooled"]
    out = df[cols].copy()
    out.columns = ["market", "n_strat", "n_pairs", "best_SR_ann",
                   "null_E[max]_ann", "DSR_best", "median_PBO", "eff_N_pooled"]
    out["best<null?"] = (out["best_SR_ann"] < out["null_E[max]_ann"]).map(
        {True: "yes", False: "no"})
    out["DSR>0.95?"] = (out["DSR_best"] > 0.95).map({True: "yes", False: "no"})
    out_disp = out.copy()
    out_disp["market"] = out_disp["market"].map(LABEL)

    out.to_csv(f"{PROJ}/tables/cross_market_summary.csv", index=False)
    with open(f"{PROJ}/tables/cross_market_summary.md", "w") as fh:
        fh.write("# Cross-market backtest-overfitting at scale "
                 "(~50k crypto + ~22.5k equity + ~20k forex strategies)\n\n")
        fh.write("All three corpora are real, causal and realistically costed "
                 "(crypto: fee/slip/funding; equity & forex via `lib/realism`: "
                 "time-of-day spreads, commission, FX rollover/triple-Wednesday "
                 "swap + weekend force-flat, equity short borrow).\n\n")
        fh.write(out_disp.round(3).to_markdown(index=False))
        fh.write("\n\n*best_SR / null are annualised; DSR uses nominal N with the "
                 "empirical trial-Sharpe dispersion (conservative). "
                 "`best<null?` = is the single best of the whole corpus below the "
                 "False-Strategy-Theorem expected max? `DSR>0.95?` = does it clear "
                 "the deflated-significance bar?*\n")
    print(out_disp.round(3).to_string(index=False))

    # ---- combined figure -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(MARKETS))
    w = 0.38
    labels = [LABEL[m] for m in MARKETS]

    # LEFT: best observed Sharpe vs False-Strategy-Theorem null, per market
    best = df["best_sr_ann"].to_numpy()
    null = df["sr0_ann"].to_numpy()
    ax = axes[0]
    ax.bar(x - w / 2, best, w, label="best observed (of corpus)",
           color=ST.PALETTE["dollar"])
    ax.bar(x + w / 2, null, w, label="E[max] under multiple-testing null",
           color=ST.PALETTE["accent"])
    for xi, (b, n) in enumerate(zip(best, null)):
        ax.text(xi - w / 2, b + 0.3 * np.sign(b) + 0.1, f"{b:.2f}",
                ha="center", va="bottom" if b >= 0 else "top", fontsize=9)
        ax.text(xi + w / 2, n + 0.2, f"{n:.1f}", ha="center", va="bottom",
                fontsize=9)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title("Corpus best vs the False-Strategy-Theorem null")
    ax.legend(fontsize=9, loc="upper left")

    # RIGHT: nominal N vs pooled effective-N (log scale -- spans 20..50000)
    nomN = df["n_strategies"].to_numpy()
    effN = df["eff_n_pooled"].to_numpy()
    ax = axes[1]
    ax.bar(x - w / 2, nomN, w, label="nominal strategies",
           color=ST.PALETTE["volume"])
    ax.bar(x + w / 2, effN, w, label="effective independent trials",
           color=ST.PALETTE["dollar"])
    ax.set_yscale("log")
    for xi, (nn, ee) in enumerate(zip(nomN, effN)):
        ax.text(xi - w / 2, nn * 1.05, f"{nn:,}", ha="center", va="bottom",
                fontsize=8)
        ax.text(xi + w / 2, ee * 1.05, f"{ee:,}", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("count (log scale)")
    ax.set_title("Nominal vs effective number of trials")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Backtest overfitting at scale across three asset classes "
                 "(real, realistically-costed corpora)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{PROJ}/figures/fig7_cross_market.png", dpi=130)
    plt.close(fig)
    print("\nwrote tables/cross_market_summary.{md,csv} and "
          "figures/fig7_cross_market.png")


if __name__ == "__main__":
    main()

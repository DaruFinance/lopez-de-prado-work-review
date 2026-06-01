#!/usr/bin/env python3
"""
Project 1 (equities, done right), Information bars on US-equity ETFs, isolating
the session/overnight-gap confound.

Two variants per ETF, matched ~daily bars from 1-min base:
  NAIVE   : all hours (incl. extended), returns across overnight gaps.
  RTH     : regular hours only (09:30-16:00 ET), within-session returns.

If dollar/tick bars look bad in NAIVE but recover in RTH, the failure was a
session/gap artifact, not a failure of LdP's method. Honest either way.

Outputs: tables/equities_session.csv, tables/equities_session.md,
         figures/fig8_equities_session_effect.png
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.exists(_os.path.join(_d, "config.py")):
    _d = _os.path.dirname(_d)
REPO_ROOT = _d
_sys.path.insert(0, REPO_ROOT)
import config as cfg
from config import LIB as _LIB
_sys.path.insert(0, _LIB)
import bars as B
import barstats as S
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETFS = [f for f in sorted(glob.glob(os.path.join(cfg.EQUITY_1M, "*.csv.gz")))
        if "_" not in os.path.basename(f).replace(".csv.gz", "")]
BAR_TYPES = ["time", "tick", "volume", "dollar"]


def run_variant(df, rth):
    days = max(50, int((df.index[-1] - df.index[0]).days))
    barset = B.matched_bars(df, n_target=days)
    rows = {}
    for bt in BAR_TYPES:
        bars = barset[bt]
        r = B.session_log_returns(bars) if rth else B.log_returns(bars)
        rows[bt] = S.return_stats(r)["exkurt"]
    return rows


def main():
    print(f"{len(ETFS)} ETFs")
    rec = []
    for f in ETFS:
        name = os.path.basename(f).replace(".csv.gz", "")
        try:
            df_all = B.load_base_equity_etf(f, rth=False)
            df_rth = B.load_base_equity_etf(f, rth=True)
        except Exception as e:
            print("  fail", name, e); continue
        if len(df_rth) < 50000:
            print("  short", name); continue
        naive = run_variant(df_all, rth=False)
        rth = run_variant(df_rth, rth=True)
        for bt in BAR_TYPES:
            rec.append(dict(etf=name, bar_type=bt,
                            exkurt_naive=naive[bt], exkurt_rth=rth[bt]))
        print(f"  {name} done")

    df = pd.DataFrame(rec)
    df.to_csv(f"{PROJ}/tables/equities_session.csv", index=False)
    summ = df.groupby("bar_type")[["exkurt_naive", "exkurt_rth"]].median().reindex(BAR_TYPES)
    with open(f"{PROJ}/tables/equities_session.md", "w") as fh:
        fh.write(f"# Equities ETFs ({df.etf.nunique()}): excess kurtosis, median across ETFs\n\n")
        fh.write("NAIVE = all hours + overnight returns; RTH = regular hours + "
                 "within-session returns. Lower is better; 0 = Gaussian.\n\n")
        fh.write(summ.round(3).to_markdown())
    print("\n=== EQUITIES: median excess kurtosis ===")
    print(summ.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(BAR_TYPES)); w = 0.38
    ax.bar(x - w/2, summ["exkurt_naive"], w, label="naive (all hrs, overnight returns)",
           color="#bbbbbb", edgecolor="white")
    ax.bar(x + w/2, summ["exkurt_rth"], w, label="RTH + within-session returns",
           color=[ST.barcolor(b) for b in BAR_TYPES], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(BAR_TYPES)
    ax.set_ylabel("Median excess kurtosis")
    ax.set_title("US equities: the dollar/tick-bar 'failure' is a session/overnight artifact\n"
                 "(handling sessions + gaps restores the information-bar advantage)")
    ax.legend()
    fig.savefig(f"{PROJ}/figures/fig8_equities_session_effect.png"); plt.close(fig)
    print("\nEquities session study done.")


if __name__ == "__main__":
    main()

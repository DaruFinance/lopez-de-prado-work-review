#!/usr/bin/env python3
"""Assemble the definitive multi-market figure + summary from saved tables.

Crypto: clean 1m (multimarket_per_pair.csv, market=crypto).
Equities: RTH + within-session returns (equities_session.csv, exkurt_rth).
Forex: tick-count clock, gap-filtered (forex_bars.csv).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/daru/ldp_review/lib")
import style as ST
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/01_information_driven_bars"
BT = ["time", "tick", "volume", "dollar"]

cr = pd.read_csv(f"{PROJ}/tables/multimarket_per_pair.csv")
cr = cr[cr.market == "crypto"].groupby("bar_type")["exkurt"].median().reindex(BT)

eq = pd.read_csv(f"{PROJ}/tables/equities_session.csv")
eq = eq.groupby("bar_type")["exkurt_rth"].median().reindex(BT)

fx = pd.read_csv(f"{PROJ}/tables/forex_bars.csv")
fx = pd.Series({"time": fx["exkurt_time"].median(), "tick": fx["exkurt_tick"].median(),
                "volume": np.nan, "dollar": np.nan})

tab = pd.DataFrame({"crypto (27 perps)": cr, "equities (9 ETFs, RTH)": eq,
                    "forex (8 majors)": fx}).reindex(BT)
tab.to_csv(f"{PROJ}/tables/FINAL_multimarket_exkurt.csv")
with open(f"{PROJ}/tables/FINAL_multimarket_exkurt.md", "w") as fh:
    fh.write("# Median excess kurtosis of bar returns — all three markets, 1-min base\n\n")
    fh.write("Crypto: clean Binance perp 1m. Equities: ETF 1m, regular hours + within-session "
             "returns. Forex: HistData tick, tick-count clock, weekend/rollover gaps dropped. "
             "Spot FX has no volume so volume/dollar bars are N/A. Lower = closer to Gaussian.\n\n")
    fh.write(tab.round(3).to_markdown())
print(tab.round(3).to_string())

markets = list(tab.columns)
x = np.arange(len(markets)); w = 0.2
fig, ax = plt.subplots(figsize=(9.5, 5))
for j, bt in enumerate(BT):
    vals = [tab.loc[bt, m] for m in markets]
    ax.bar(x + (j - 1.5) * w, np.nan_to_num(vals, nan=0.0), w, color=ST.barcolor(bt),
           label=bt, alpha=0.88, edgecolor="white")
    for i, v in enumerate(vals):
        if np.isnan(v):
            ax.text(x[i] + (j - 1.5) * w, 0.3, "N/A", ha="center", va="bottom",
                    fontsize=7, color="gray", rotation=90)
ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(markets)
ax.set_ylabel("Median excess kurtosis of bar returns")
ax.set_title("Information-driven bars Gaussianize returns across Crypto, Equities & Forex\n"
             "(1-minute base; 0 = Gaussian; lower is better. Equities require session-aware handling; "
             "spot FX uses the tick clock)")
ax.legend(title="bar type", ncol=4, loc="upper right")
fig.savefig(f"{PROJ}/figures/FINAL_multimarket_kurtosis.png")
print("\nFINAL figure + table written")

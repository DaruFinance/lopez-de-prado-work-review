#!/usr/bin/env python3
"""
Project 1b, Does the information-bar advantage depend on BASE granularity?

LdP builds information bars from raw ticks. Practitioners usually build them by
accumulating coarser bricks (1m, 5m, 30m...). Hypothesis: the Gaussianizing
benefit of dollar/volume bars over time bars *shrinks and can reverse* as the
base brick gets coarser, because a single coarse brick can hold a whole bar's
worth of activity, making information-bar horizons wildly heterogeneous.

For each pair we hold the TARGET bar frequency fixed (~daily) and vary only the
base granularity (1, 5, 15, 30, 60 min), all resampled from the SAME 1m data,
then measure excess kurtosis of each bar type's returns.

Outputs: tables/granularity.csv, figures/fig5_granularity_kurtosis.png,
         figures/fig6_dollar_minus_time_gap.png, tables/count_stability_infobars.md
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
CACHE = cfg.CRYPTO_1M
GRANS = [1, 5, 15, 30, 60]            # base brick size in minutes
BAR_TYPES = ["time", "tick", "volume", "dollar"]


def load_1m(path):
    df = pd.read_parquet(path)
    df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns=["open_time"])
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]
    return df


def resample_base(df1m, minutes):
    if minutes == 1:
        return df1m
    g = df1m.resample(f"{minutes}min", label="right", closed="right")
    base = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "quote_volume": g["quote_volume"].sum(),
        "count": g["count"].sum(),
        "taker_buy_volume": g["taker_buy_volume"].sum(),
        "taker_buy_quote_volume": g["taker_buy_quote_volume"].sum(),
    }).dropna()
    return base[base["count"] > 0]


def main():
    files = sorted(glob.glob(f"{CACHE}/*_1m.parquet"))
    if not files:
        print("No 1m cache yet at", CACHE); sys.exit(1)
    print("pairs:", [os.path.basename(f).replace('_1m.parquet','') for f in files])

    rows, stab_rows = [], []
    for f in files:
        name = os.path.basename(f).replace("_1m.parquet", "")
        df1m = load_1m(f)
        days = max(50, int((df1m.index[-1] - df1m.index[0]).days))
        for g in GRANS:
            base = resample_base(df1m, g)
            if len(base) < 2000:
                continue
            barset = B.matched_bars(base, n_target=days)   # fixed ~daily target
            rec = dict(pair=name, gran_min=g, base_bars=len(base))
            for bt in BAR_TYPES:
                st = S.return_stats(B.log_returns(barset[bt]))
                rec[f"exkurt_{bt}"] = st["exkurt"]
                rec[f"absskew_{bt}"] = abs(st["skew"]) if st["skew"] == st["skew"] else np.nan
                if g == 1:   # count stability only meaningful among info bars; report at 1m
                    cs = S.count_stability(barset[bt], "W")
                    stab_rows.append(dict(pair=name, bar_type=bt, count_cv=cs["cv"]))
            rows.append(rec)
        print(f"  {name}: done ({days} days)")

    res = pd.DataFrame(rows)
    res.to_csv(f"{PROJ}/tables/granularity.csv", index=False)

    # corrected count-stability: information bars only (time bars excluded)
    stab = pd.DataFrame(stab_rows)
    info = stab[stab.bar_type != "time"].groupby("bar_type")["count_cv"].median().reindex(["tick","volume","dollar"])
    with open(f"{PROJ}/tables/count_stability_infobars.md", "w") as fh:
        fh.write("# Bar-count stability among INFORMATION bars (1m base, median CV)\n\n")
        fh.write("Time bars excluded (their CV is ~0 by construction). LdP claims "
                 "dollar bars are the most stable.\n\n")
        fh.write(info.round(4).to_markdown())
    print("\ncount-stability (info bars, 1m base, median CV):\n", info.round(4).to_string())

    make_figs(res)
    print("\nGranularity study done.")


def make_figs(res):
    figd = f"{PROJ}/figures"

    # Fig 5, median excess kurtosis vs base granularity, per bar type
    med = res.groupby("gran_min")[[f"exkurt_{b}" for b in BAR_TYPES]].median()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for bt in BAR_TYPES:
        ax.plot(med.index, med[f"exkurt_{bt}"], "o-", color=ST.barcolor(bt),
                label=bt, lw=2, ms=6)
    ax.set_xscale("log"); ax.set_xticks(GRANS); ax.set_xticklabels(GRANS)
    ax.set_xlabel("Base brick granularity (minutes, log scale)")
    ax.set_ylabel("Median excess kurtosis of bar returns")
    ax.set_title("The information-bar advantage is granularity-dependent\n"
                 "(finer base data → dollar/volume bars Gaussianize; coarse base → they don't)")
    ax.legend(title="bar type")
    fig.savefig(f"{figd}/fig5_granularity_kurtosis.png"); plt.close(fig)

    # Fig 6, dollar-minus-time kurtosis gap vs granularity (crossing zero)
    res["gap_dollar"] = res["exkurt_dollar"] - res["exkurt_time"]
    res["gap_volume"] = res["exkurt_volume"] - res["exkurt_time"]
    gap = res.groupby("gran_min")[["gap_dollar", "gap_volume"]].median()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(gap.index, gap["gap_dollar"], "o-", color=ST.barcolor("dollar"),
            label="dollar − time", lw=2, ms=6)
    ax.plot(gap.index, gap["gap_volume"], "s-", color=ST.barcolor("volume"),
            label="volume − time", lw=2, ms=6)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.fill_between(gap.index, gap.min().min()*1.1, 0, color="#009E73", alpha=0.06)
    ax.set_xscale("log"); ax.set_xticks(GRANS); ax.set_xticklabels(GRANS)
    ax.set_xlabel("Base brick granularity (minutes, log scale)")
    ax.set_ylabel("Excess-kurtosis gap vs time bars")
    ax.set_title("Below zero = information bars beat time bars on tail-heaviness")
    ax.legend()
    fig.savefig(f"{figd}/fig6_dollar_minus_time_gap.png"); plt.close(fig)
    print("granularity figures written")


if __name__ == "__main__":
    main()

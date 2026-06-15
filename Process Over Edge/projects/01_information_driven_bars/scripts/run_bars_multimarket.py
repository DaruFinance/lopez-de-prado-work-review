#!/usr/bin/env python3
"""
Project 1 (multi-market), Information-driven bars across CRYPTO and US EQUITIES.

Crypto: clean 1m Binance USD-M perp dumps (2022-2024), 24 pairs.
Equities: Algoseek ETF 1-min (SPY/QQQ/IWM + sector SPDRs + vol), deep history.
Both have Volume + trade count, so time/tick/volume/dollar bars are all built
from a 1-minute base at a matched ~daily frequency, then compared on the
statistical properties LdP cares about (Gaussianity, serial correlation).

Forex is handled separately (no native volume, see writeup / data note).

Outputs: tables/multimarket_per_pair.csv, tables/multimarket_summary.md,
         figures/fig7_multimarket_kurtosis.png
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

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
CRYPTO = sorted(glob.glob(os.path.join(cfg.CRYPTO_1M, "*_1m.parquet")))
EQUITY = [f for f in sorted(glob.glob(os.path.join(cfg.EQUITY_1M, "*.csv.gz")))
          if "_" not in os.path.basename(f).replace(".csv.gz", "")]  # combined files only
BAR_TYPES = ["time", "tick", "volume", "dollar"]


def load_crypto(path):
    df = pd.read_parquet(path)
    df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns=["open_time"])
    return df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]


def process(task):
    market, path = task
    name = os.path.basename(path).split("_")[0].replace(".csv.gz", "").replace(".parquet", "")
    try:
        df = load_crypto(path) if market == "crypto" else B.load_base_equity_etf(path)
    except Exception as e:
        print("  load fail", name, e); return None
    if len(df) < 50000:
        return None
    days = max(50, int((df.index[-1] - df.index[0]).days))
    barset = B.matched_bars(df, n_target=days)
    rows = []
    for bt in BAR_TYPES:
        st = S.return_stats(B.log_returns(barset[bt]))
        cs = S.count_stability(barset[bt], "W")
        rows.append(dict(market=market, pair=name, bar_type=bt, n_bars=len(barset[bt]),
                         skew=st["skew"], exkurt=st["exkurt"], jb=st["jb"],
                         jb_p=st["jb_p"], ac1=st["ac1"], count_cv=cs["cv"]))
    print(f"  {market}:{name} done")
    return pd.DataFrame(rows)


def main():
    tasks = [("crypto", f) for f in CRYPTO] + [("equity", f) for f in EQUITY]
    print(f"{len(CRYPTO)} crypto + {len(EQUITY)} equity ETFs")
    with Pool(min(12, len(tasks))) as pool:
        out = [r for r in pool.map(process, tasks) if r is not None]
    df = pd.concat(out, ignore_index=True)
    df.to_csv(f"{PROJ}/tables/multimarket_per_pair.csv", index=False)

    summ = (df.assign(abs_skew=df["skew"].abs(), abs_ac1=df["ac1"].abs())
            .groupby(["market", "bar_type"])
            .agg(n=("pair", "nunique"),
                 med_exkurt=("exkurt", "median"),
                 med_abs_skew=("abs_skew", "median"),
                 med_abs_ac1=("abs_ac1", "median"),
                 frac_normal=("jb_p", lambda s: float((s > 0.05).mean())))
            .reindex(pd.MultiIndex.from_product([["crypto", "equity"], BAR_TYPES],
                                                names=["market", "bar_type"])))
    with open(f"{PROJ}/tables/multimarket_summary.md", "w") as fh:
        fh.write("# Information-driven bars, Crypto vs US Equities (median across instruments)\n\n")
        fh.write(f"Crypto: {df[df.market=='crypto'].pair.nunique()} Binance perps (1m, 2022-2024). "
                 f"Equities: {df[df.market=='equity'].pair.nunique()} Algoseek ETFs (1m). "
                 "Matched ~daily bars. Lower excess kurtosis / |skew| / |AC(1)| is better.\n\n")
        fh.write(summ.round(4).to_markdown())
    print("\n=== MULTI-MARKET SUMMARY ===")
    print(summ.round(4).to_string())

    # Fig 7, excess kurtosis by bar type, grouped by market
    fig, ax = plt.subplots(figsize=(9, 4.6))
    markets = ["crypto", "equity"]
    width = 0.18
    xc = np.arange(len(markets))
    for j, bt in enumerate(BAR_TYPES):
        vals = [df[(df.market == m) & (df.bar_type == bt)]["exkurt"].median() for m in markets]
        ax.bar(xc + (j - 1.5) * width, vals, width, color=ST.barcolor(bt),
               label=bt, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xc); ax.set_xticklabels(["Crypto\n(24 perps)", "US Equities\n(9 ETFs)"])
    ax.set_ylabel("Median excess kurtosis of bar returns")
    ax.set_title("Information-driven bars Gaussianize returns in BOTH markets\n"
                 "(0 = Gaussian; lower is better; built from 1-minute base)")
    ax.legend(title="bar type", ncol=4)
    fig.savefig(f"{PROJ}/figures/fig7_multimarket_kurtosis.png"); plt.close(fig)
    print("\nMulti-market figure written.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Project 0, Backtest Overfitting & the Deflated Sharpe Ratio, at scale across
Crypto / US Equities / Forex.

The LdP setup: optimise a structural strategy family (dual moving-average
crossover) over a grid of parameters on the SAME real series. Each parameter
combo is one "trial". We then ask his questions:
  - How impressive is the best in-sample Sharpe, and how does it compare to the
    expected max Sharpe of skill-less trials (False Strategy Theorem)?
  - What is the Probability of Backtest Overfitting (PBO, via CSCV)?
  - Does the Deflated Sharpe Ratio survive once we correct for the number and
    correlation of trials?
  - How many *effectively independent* trials were there (ONC clustering)?

Bars are information-driven (dogfooding Project 1): dollar bars for crypto &
equities (equities within regular hours), tick bars for forex. All returns are
NET OF COSTS (no costless backtests).

Outputs: tables/overfit_per_instrument.csv, tables/overfit_summary.md,
         figures/fig1_is_vs_oos.png, fig2_maxsr_vs_null.png,
         fig3_dsr_by_market.png, fig4_effective_trials.png
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

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
import overfit as OF
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTO_CACHE = cfg.CRYPTO_1M
FX_CACHE = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M

# per-unit-turnover cost (round-trip handled via |position change|), realistic & non-zero
COST = {"crypto": 0.0007, "equity": 0.0002, "forex": 0.0001}
FAST = [2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 50]
SLOW = [20, 28, 40, 55, 75, 100, 130, 170, 220, 290, 380, 500]
BARS_PER_DAY = 8                       # ~ 3-hourly information bars


def ma_crossover_matrix(bars, cost):
    """Return (T x N) matrix of per-bar net returns for the (fast,slow) grid."""
    close = bars["close"].to_numpy()
    barret = np.diff(np.log(close), prepend=np.log(close[0]))   # per-bar log return
    cache = {}
    def ma(w):
        if w not in cache:
            cache[w] = pd.Series(close).rolling(w).mean().to_numpy()
        return cache[w]
    cols, names = [], []
    for f in FAST:
        for s in SLOW:
            if f >= s:
                continue
            pos = np.sign(ma(f) - ma(s))            # +1/-1, decided at bar close
            pos = np.nan_to_num(pos)
            held = np.roll(pos, 1); held[0] = 0     # hold over next bar (causal)
            turn = np.abs(np.diff(pos, prepend=0))
            ret = held * barret - cost * turn
            cols.append(ret); names.append((f, s))
    return np.column_stack(cols), names, barret


def load_instrument(market, path):
    if market == "crypto":
        df = pd.read_parquet(path)
        df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns="open_time")
        df = df[(df.close > 0) & (df.volume > 0) & (df["count"] > 0)]
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.matched_bars(df, days * BARS_PER_DAY)["dollar"]
    if market == "equity":
        df = B.load_base_equity_etf(path, rth=True)
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.matched_bars(df, days * 6)["dollar"]
    if market == "forex":
        df = pd.read_parquet(path)
        df["volume"] = df["count"]; df["quote_volume"] = df["count"]
        df["taker_buy_volume"] = df["count"] / 2; df["taker_buy_quote_volume"] = df["count"] / 2
        df = df[(df.close > 0) & (df["count"] > 0)]
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.threshold_bars(df, "count", df["count"].sum() / (days * BARS_PER_DAY))


INSTR = {
    "crypto": [f"{CRYPTO_CACHE}/{p}_1m.parquet" for p in
               ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT", "LTCUSDT", "LINKUSDT"]],
    "equity": [f"{ETF_DIR}/{p}.csv.gz" for p in ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]],
    "forex":  [f"{FX_CACHE}/{p}_fx1m.parquet" for p in
               ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURGBP"]],
}


def analyse(market, path):
    name = os.path.basename(path).split("_")[0].replace(".csv.gz", "").replace(".parquet", "")
    bars = load_instrument(market, path)
    if bars is None or len(bars) < 800:
        return None
    M, names, _ = ma_crossover_matrix(bars, COST[market])
    M = M[10:]                                   # drop warm-up rows with NaNs
    M = M[: np.all(np.isfinite(M), axis=0)]
    T, N = M.shape
    sr = M.mean(0) / M.std(0, ddof=1)            # per-bar Sharpe of each trial
    best = int(np.argmax(sr))
    rb = M[: best]
    ann = np.sqrt(BARS_PER_DAY * 252)            # annualisation factor for display
    d = OF.deflated_sharpe_ratio(OF.sharpe(rb), T, ss.skew(rb),
                                 ss.kurtosis(rb, fisher=False), sr)
    pbo = OF.pbo_cscv(M, 12)
    eff = OF.effective_n_trials(M)
    return dict(market=market, instrument=name, T=T, N_trials=N,
                best_sr_ann=sr.max() * ann, sr0_ann=d["sr0"] * ann,
                dsr=d["dsr"], pbo=pbo["pbo"],
                eff_trials=eff["effective_n"], corr_med=eff["corr_med"],
                _M=M, _sr=sr, _best=best)


def main():
    rows, keep = [], {}
    for market, paths in INSTR.items():
        for p in paths:
            if not os.path.exists(p):
                print("  missing", p); continue
            try:
                r = analyse(market, p)
            except Exception as e:
                print("  fail", p, e); continue
            if r is None:
                continue
            keep.setdefault(market, r)            # stash one per market for plots
            print(f"  {market}:{r['instrument']} N={r['N_trials']} bestSR(ann)={r['best_sr_ann']:.2f} "
                  f"DSR={r['dsr']:.3f} PBO={r['pbo']:.2f} effN={r['eff_trials']}")
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    df = pd.DataFrame(rows)
    df.to_csv(f"{PROJ}/tables/overfit_per_instrument.csv", index=False)
    summ = df.groupby("market").agg(
        instruments=("instrument", "nunique"), trials=("N_trials", "median"),
        best_sr_ann=("best_sr_ann", "median"), sr0_ann=("sr0_ann", "median"),
        dsr=("dsr", "median"), pbo=("pbo", "median"),
        eff_trials=("eff_trials", "median")).reindex(["crypto", "equity", "forex"])
    with open(f"{PROJ}/tables/overfit_summary.md", "w") as fh:
        fh.write("# Backtest overfitting at scale, median by market\n\n")
        fh.write("MA-crossover grid optimised per instrument (each combo = one trial), "
                 "net of costs, on information-driven bars. best_sr_ann = best in-sample "
                 "annualised Sharpe; sr0_ann = expected-max Sharpe of skill-less trials "
                 "(False Strategy Theorem); DSR = deflated Sharpe (prob the winner is real); "
                 "PBO = prob. of backtest overfitting; eff_trials = independent trials (ONC).\n\n")
        fh.write(summ.round(3).to_markdown())
    print("\n=== SUMMARY BY MARKET ===\n", summ.round(3).to_string())
    make_figs(df, keep)
    print("\nProject 0 at-scale done.")


def make_figs(df, keep):
    d = f"{PROJ}/figures"

    # Fig 1, IS vs OOS Sharpe scatter (overfitting), representative crypto instrument
    r = keep.get("crypto") or next(iter(keep.values()))
    M = r["_M"]; T = M.shape[0]; half = T // 2
    sr_is = M[:half].mean(0) / M[:half].std(0, ddof=1)
    sr_oos = M[half:].mean(0) / M[half:].std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(sr_is, sr_oos, s=10, alpha=0.5, color=ST.PALETTE["dollar"])
    bi = int(np.argmax(sr_is))
    ax.scatter([sr_is[bi]], [sr_oos[bi]], s=90, color=ST.PALETTE["accent"],
               zorder=5, label="best in-sample")
    lim = max(abs(sr_is).max(), abs(sr_oos).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("In-sample Sharpe (per bar)"); ax.set_ylabel("Out-of-sample Sharpe (per bar)")
    ax.set_title(f"Overfitting: high in-sample Sharpe does not carry OOS\n({r['instrument']}, "
                 f"{M.shape[1]} MA-crossover trials)")
    ax.legend()
    fig.savefig(f"{d}/fig1_is_vs_oos.png"); plt.close(fig)

    # Fig 2, best IS Sharpe vs False-Strategy-Theorem null, by market
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    g = df.groupby("market")[["best_sr_ann", "sr0_ann"]].median().reindex(["crypto", "equity", "forex"])
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w/2, g["best_sr_ann"], w, label="best in-sample Sharpe (annualised)", color=ST.PALETTE["dollar"])
    ax.bar(x + w/2, g["sr0_ann"], w, label="expected-max under the null (False Strategy Thm)", color="#bbbbbb")
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylabel("Annualised Sharpe")
    ax.set_title("The 'impressive' best backtest is within what pure luck produces")
    ax.legend()
    fig.savefig(f"{d}/fig2_maxsr_vs_null.png"); plt.close(fig)

    # Fig 3, DSR and PBO by instrument
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, col, ttl, ref in [(axes[0], "dsr", "Deflated Sharpe Ratio\n(prob. the winner is real; <0.95 = not significant)", 0.95),
                              (axes[1], "pbo", "Probability of Backtest Overfitting\n(>0.5 = overfit)", 0.5)]:
        for i, m in enumerate(["crypto", "equity", "forex"]):
            vals = df.loc[df.market == m, col].dropna()
            ax.scatter([i] * len(vals), vals, s=40, alpha=0.7, color=ST.PALETTE["dollar"])
        ax.axhline(ref, color=ST.PALETTE["accent"], ls="--", lw=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(["crypto", "equity", "forex"])
        ax.set_title(ttl)
    fig.savefig(f"{d}/fig3_dsr_pbo.png"); plt.close(fig)

    # Fig 4, nominal vs effective number of trials
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    g = df.groupby("market")[["N_trials", "eff_trials"]].median().reindex(["crypto", "equity", "forex"])
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w/2, g["N_trials"], w, label="nominal trials (grid size)", color="#bbbbbb")
    ax.bar(x + w/2, g["eff_trials"], w, label="effectively independent (ONC)", color=ST.PALETTE["volume"])
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylabel("Number of trials")
    ax.set_title("Hundreds of parameter combos are only a handful of independent bets")
    ax.legend()
    fig.savefig(f"{d}/fig4_effective_trials.png"); plt.close(fig)
    print("figures written:", os.listdir(d))


if __name__ == "__main__":
    main()

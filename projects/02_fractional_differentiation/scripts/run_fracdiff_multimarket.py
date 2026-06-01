#!/usr/bin/env python3
"""
Project 2, Fractional Differentiation, MULTI-MARKET extension.

Extends the completed crypto FFD study (Lopez de Prado, AFML Ch.5) to two more
markets so d* is comparable across CRYPTO vs US EQUITIES vs FOREX, all at 1h.

Methodology is MIRRORED EXACTLY from run_fracdiff_study.py so cross-market
numbers are comparable:
  * Apply FFD to log(close), causal only (no lookahead).
  * SAME d grid:  D_GRID = arange(0.0, 1.0001, 0.05)
  * SAME tau:     TAU = 1e-5
  * SAME ADF logic: statsmodels adfuller(maxlag=1, regression="c", autolag=None);
    d* = smallest d on the grid whose FFD series passes ADF at the 5% crit value
    (via lib.fracdiff.min_d_search, identical to the crypto run).
  * SAME memory metric: Pearson corr( FFD series , log-price level ) on the
    overlapping non-NaN support, reported at d* and at d=1 (plain returns).
  * SAME MIN_ROWS = 5000 minimum-length gate.

Crypto per-pair results are LOADED from tables/per_pair_fracdiff.csv (NOT
recomputed). This script only computes equities (9 Algoseek ETFs resampled to
1h) and forex (3 FXCM pairs at 1h), then combines all three markets.

Standalone & idempotent.  Run: python3 run_fracdiff_multimarket.py
"""
import sys, os, glob, warnings
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
import fracdiff as F
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

# --------------------------------------------------------------------------- #
# SHARED PARAMETERS, identical to run_fracdiff_study.py
# --------------------------------------------------------------------------- #
MIN_ROWS = 5000
TAU = 1e-5
D_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
LDP_BAND = (0.30, 0.60)            # LdP's reported d* range for equities/FX

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "figures")
TAB = os.path.join(PROJ, "tables")
CRYPTO_PER_PAIR = os.path.join(TAB, "per_pair_fracdiff.csv")

ETF_DIR = cfg.EQUITY_1M
ETF_TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "VXX", "UVXY"]
FOREX_FILES = {
    "EURUSD": (os.path.join(cfg.FX_RAW, "EURUSD_FXCM.csv"), True),   # (path, needs 1h resample)
    "EURGBP": (os.path.join(cfg.FX_RAW, "EURGBP_h1.csv"), False),
    "USDJPY": (os.path.join(cfg.FX_RAW, "USDJPY_h1_2016_2026.csv"), False),
}

os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


# --------------------------------------------------------------------------- #
# data loaders  -> return log(close) as a clean float array (mirror crypto)
# --------------------------------------------------------------------------- #
def _clean_logclose(close):
    """Mirror crypto load_logclose: drop non-finite, non-positive, flat tail, log."""
    c = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy()
    c = c[np.isfinite(c)]
    c = c[c > 0]
    if len(c) > 10:
        i = len(c) - 1
        while i > 0 and c[i] == c[i - 1]:
            i -= 1
        c = c[: i + 1]
    return np.log(c)


def load_etf_1h(ticker):
    """Combined Algoseek 1-min ETF file -> 1h close series (last price/hour)."""
    path = os.path.join(ETF_DIR, f"{ticker}.csv.gz")
    df = pd.read_csv(path, usecols=["BarDateTime", "LastTradePrice"],
                     compression="gzip")
    t = pd.to_datetime(df["BarDateTime"], utc=True, errors="coerce")
    s = pd.Series(pd.to_numeric(df["LastTradePrice"], errors="coerce").values,
                  index=t).dropna().sort_index()
    h = s.resample("1h").last().dropna()        # last trade price in each hour, causal
    return _clean_logclose(h.values)


def load_forex_1h(path, needs_resample):
    df = pd.read_csv(path, usecols=["time", "close"])
    t = pd.to_datetime(df["time"], unit="s", utc=True)
    s = pd.Series(pd.to_numeric(df["close"], errors="coerce").values,
                  index=t).dropna().sort_index()
    if needs_resample:
        s = s.resample("1h").last().dropna()    # sub-hour -> 1h last price, causal
    return _clean_logclose(s.values)


# --------------------------------------------------------------------------- #
# per-instrument FFD (identical analysis to crypto process_pair)
# --------------------------------------------------------------------------- #
def analyse(market, instrument, x):
    if len(x) < MIN_ROWS:
        return {"market": market, "instrument": instrument, "n": len(x),
                "d_star": np.nan, "adf_stat": np.nan, "adf_p": np.nan,
                "corr_level_dstar": np.nan, "corr_level_d1": np.nan,
                "error": f"short {len(x)}"}
    res = F.min_d_search(x, d_grid=D_GRID, tau=TAU)
    star = res["star"]
    trace = {r["d"]: r for r in res["trace"]}
    r1 = trace.get(1.0, {})
    if star is None:
        return {"market": market, "instrument": instrument, "n": len(x),
                "d_star": np.nan, "adf_stat": np.nan, "adf_p": np.nan,
                "corr_level_dstar": np.nan,
                "corr_level_d1": r1.get("corr_level", np.nan),
                "error": "no_dstar"}
    return {"market": market, "instrument": instrument, "n": len(x),
            "d_star": res["d_star"], "adf_stat": star["adf_stat"],
            "adf_p": star["adf_p"], "corr_level_dstar": star["corr_level"],
            "corr_level_d1": r1.get("corr_level", np.nan), "error": ""}


# --------------------------------------------------------------------------- #
# build the combined per-instrument frame
# --------------------------------------------------------------------------- #
def build_table():
    rows = []

    # --- crypto: LOAD existing per-pair results (do not recompute) ---
    cr = pd.read_csv(CRYPTO_PER_PAIR)
    for _, r in cr.iterrows():
        rows.append({"market": "crypto", "instrument": r["pair"],
                     "d_star": r["d_star"], "adf_stat": r["adf_stat_at_dstar"],
                     "adf_p": r["adf_p_at_dstar"],
                     "corr_level_dstar": r["corr_level_at_dstar"],
                     "corr_level_d1": r["corr_level_at_d1"]})
    print(f"  crypto: loaded {len(cr)} pairs from {os.path.basename(CRYPTO_PER_PAIR)}")

    # --- equities: 9 ETFs resampled to 1h ---
    for tk in ETF_TICKERS:
        try:
            x = load_etf_1h(tk)
            r = analyse("equities", tk, x)
        except Exception as e:
            r = {"market": "equities", "instrument": tk, "d_star": np.nan,
                 "adf_stat": np.nan, "adf_p": np.nan, "corr_level_dstar": np.nan,
                 "corr_level_d1": np.nan, "error": f"load {e}"}
        print(f"  equities {tk:5s}: n={r.get('n','?')} d*={r['d_star']} "
              f"corr_dstar={r['corr_level_dstar']} {r.get('error','')}")
        rows.append({k: r.get(k) for k in
                     ["market", "instrument", "d_star", "adf_stat", "adf_p",
                      "corr_level_dstar", "corr_level_d1"]})

    # --- forex: 3 pairs at 1h ---
    for name, (path, rs) in FOREX_FILES.items():
        try:
            x = load_forex_1h(path, rs)
            r = analyse("forex", name, x)
        except Exception as e:
            r = {"market": "forex", "instrument": name, "d_star": np.nan,
                 "adf_stat": np.nan, "adf_p": np.nan, "corr_level_dstar": np.nan,
                 "corr_level_d1": np.nan, "error": f"load {e}"}
        print(f"  forex {name:7s}: n={r.get('n','?')} d*={r['d_star']} "
              f"corr_dstar={r['corr_level_dstar']} {r.get('error','')}")
        rows.append({k: r.get(k) for k in
                     ["market", "instrument", "d_star", "adf_stat", "adf_p",
                      "corr_level_dstar", "corr_level_d1"]})

    df = pd.DataFrame(rows, columns=["market", "instrument", "d_star",
                                     "adf_stat", "adf_p", "corr_level_dstar",
                                     "corr_level_d1"])
    df.to_csv(os.path.join(TAB, "multimarket_fracdiff_per_instrument.csv"),
              index=False)
    return df


# --------------------------------------------------------------------------- #
# summary markdown by market
# --------------------------------------------------------------------------- #
def write_summary(df):
    order = ["crypto", "equities", "forex"]
    lines = ["# Fractional Differentiation, multi-market summary (1h, FFD tau=1e-5)\n",
             "Same d grid [0,1] step 0.05, same ADF logic, same memory metric "
             "(corr of FFD series with the log-price level) across all markets.\n",
             "\n| market | n | median d* | IQR (Q1-Q3) | % d*<1 | median corr@d* | median |corr|@d=1 |",
             "|---|---|---|---|---|---|---|"]
    stats = {}
    for m in order:
        sub = df[df["market"] == m]
        ds = sub["d_star"].dropna()
        if len(ds) == 0:
            continue
        q1, med, q3 = ds.quantile([.25, .5, .75])
        frac = (ds < 1.0).mean()
        md = sub["corr_level_dstar"].dropna().median()
        m1 = sub["corr_level_d1"].dropna().abs().median()
        stats[m] = dict(n=len(ds), med=med, q1=q1, q3=q3, frac=frac, md=md, m1=m1)
        lines.append(f"| {m} | {len(ds)} | {med:.3f} | {q1:.3f}-{q3:.3f} | "
                     f"{frac:.0%} | {md:.3f} | {m1:.3f} |")

    lines.append(f"\nLdP reference band for equities/FX: d* ~ {LDP_BAND[0]:.1f}-{LDP_BAND[1]:.1f}.\n")
    lines.append("\n## Per-instrument (equities + forex)\n")
    sub = df[df["market"].isin(["equities", "forex"])][
        ["market", "instrument", "d_star", "adf_stat", "adf_p",
         "corr_level_dstar", "corr_level_d1"]].copy()
    lines.append(sub.round(4).to_markdown(index=False))
    with open(os.path.join(TAB, "multimarket_fracdiff_summary.md"), "w") as f:
        f.write("\n".join(lines))
    return stats


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_dstar_by_market(df, stats):
    order = ["crypto", "equities", "forex"]
    cols = {"crypto": ST.PALETTE["dollar"], "equities": ST.PALETTE["tick"],
            "forex": ST.PALETTE["time"]}
    data = [df[df["market"] == m]["d_star"].dropna().values for m in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    # LdP reference band
    ax.axhspan(LDP_BAND[0], LDP_BAND[1], color=ST.PALETTE["accent"], alpha=0.12,
               zorder=0)
    ax.text(3.45, np.mean(LDP_BAND), "LdP eq/FX\n~0.3-0.6", fontsize=8,
            color=ST.PALETTE["accent"], va="center", ha="left")

    bp = ax.boxplot(data, positions=range(1, 4), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#222", lw=2))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(cols[m]); patch.set_alpha(0.45)

    rng = np.random.default_rng(0)
    for i, (m, vals) in enumerate(zip(order, data), start=1):
        if len(vals) == 0:
            continue
        jit = (rng.random(len(vals)) - 0.5) * 0.28
        ax.scatter(np.full(len(vals), i) + jit, vals, s=14, color=cols[m],
                   alpha=0.5, edgecolor="none", zorder=3)
        med = stats[m]["med"]
        ax.annotate(f"med {med:.2f}", xy=(i, med), xytext=(i + 0.30, med),
                    fontsize=9, fontweight="bold", color="#222", va="center")

    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([f"{m}\n(n={stats.get(m,{}).get('n',0)})" for m in order])
    ax.set_ylabel("minimum d* for ADF stationarity")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.4, 3.9)
    ax.axhline(1.0, ls=":", color="#999", lw=1.0)
    ax.set_title("Minimum fractional exponent d* by market (1h, FFD)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_dstar_by_market.png"), dpi=200)
    plt.close(fig)


def fig_memory_by_market(df, stats):
    order = ["crypto", "equities", "forex"]
    x = np.arange(len(order))
    w = 0.38
    md = [stats[m]["md"] for m in order]
    m1 = [stats[m]["m1"] for m in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar(x - w / 2, md, w, color=ST.PALETTE["dollar"], alpha=0.85,
                label="FFD at d*  (memory kept)")
    b2 = ax.bar(x + w / 2, m1, w, color=ST.PALETTE["time"], alpha=0.85,
                label="returns d=1  (|corr|)")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}", xy=(b.get_x() + b.get_width() / 2,
                    b.get_height()), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={stats[m]['n']})" for m in order])
    ax.set_ylabel("median  corr( differenced series , log-price level )")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    ax.set_title("Memory retained: FFD(d*) preserves the level, returns (d=1) erase it")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_memory_by_market.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print("[1/3] building multi-market per-instrument table ...")
    df = build_table()
    print("[2/3] writing summary ...")
    stats = write_summary(df)
    print("[3/3] figures ...")
    fig_dstar_by_market(df, stats)
    fig_memory_by_market(df, stats)

    print("\n=== by-market median d* ===")
    for m in ["crypto", "equities", "forex"]:
        if m in stats:
            s = stats[m]
            print(f"  {m:9s} n={s['n']:>4d}  median d*={s['med']:.3f}  "
                  f"%d*<1={s['frac']:.0%}  mem@d*={s['md']:.3f}  |mem|@d1={s['m1']:.3f}")
    print("done.")


if __name__ == "__main__":
    main()

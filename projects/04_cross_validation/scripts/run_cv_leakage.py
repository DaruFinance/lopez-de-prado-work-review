#!/usr/bin/env python3
"""
Project 04, Cross-Validation in Finance: standard k-fold LEAKS; purging+embargo
fixes it; CPCV gives a distribution of OOS paths.  (López de Prado, AFML Ch.7 & 12)

THE CLAIM (LdP).  In finance, labels are built from windows of future bars (a
fixed-horizon return over H bars, or a triple-barrier label with max-holding H).
Consecutive labels therefore SHARE information: label_t and label_{t+1} both look
at the same future bars.  Standard k-fold cross-validation places such
overlapping points on both sides of the train/test cut, so the test fold is
contaminated by training information, the CV score is optimistically biased.
PURGING removes train points whose label window overlaps the test fold; an
EMBARGO drops a few train points right after the test fold to kill the residual
serial correlation.  COMBINATORIAL PURGED CV (CPCV) holds out every combination
of k_test groups, yielding a DISTRIBUTION of OOS path scores (a single backtest
is just one draw from it).

THE EXPERIMENT.  On real bars (crypto dollar bars, US-equity ETF dollar bars,
forex tick bars) we build fixed-horizon overlapping labels with overlap span H,
train a bagged-tree classifier (RandomForest) on strictly-causal features, and
score it THREE ways:
   (a) standard sklearn KFold            -> leaky
   (b) purged k-fold + embargo           -> clean
   (c) CPCV                              -> distribution of OOS paths
We report the INFLATION = (leaky score) - (purged score), show it GROWS with the
label overlap H and SHRINKS with the embargo, and plot the CPCV OOS distribution
with the single-split "backtest" marked.

CAUSALITY.  Every feature at bar t uses only bars <= t.  The label at bar t uses
bars t+1..t+H (strictly future) and is the only forward-looking object, exactly
what cross-validation must protect.

PROFILE-THEN-NUMBA.  A cProfile smoke test (--profile) shows RandomForest.fit
dominates wall time (>90%).  Per house rules we do NOT Numba sklearn.  The one
non-sklearn hot loop is label construction; it is moved to an @njit kernel and
VERIFIED bit-identical against a pure-python reference (--verify, prints max|Δ|).

Outputs (idempotent):
  tables/cv_per_instrument.csv, cv_by_market.md/.csv,
          cv_inflation_vs_overlap.csv, cv_inflation_vs_embargo.csv
  figures/fig1_kfold_vs_purged_by_market.png
          fig2_inflation_vs_overlap.png
          fig3_inflation_vs_embargo.png
          fig4_cpcv_distribution.png

Rerun:
  python3 run_cv_leakage.py --verify      # bit-identical label-kernel check
  python3 run_cv_leakage.py --profile     # cProfile a single-instrument smoke test
  python3 run_cv_leakage.py               # full multi-market run
"""
from __future__ import annotations
import sys, os, glob, argparse, warnings, time
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
import fracdiff as FD

from numba import njit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTO_CACHE = cfg.CRYPTO_1M
FX_CACHE = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M

BARS_PER_DAY = 8                 # ~3-hourly information bars (matches Project 0/1)
N_SPLITS = 6                     # k for k-fold / purged k-fold
N_TREES = 200
RF_KW = dict(n_estimators=N_TREES, max_depth=5, min_samples_leaf=50,
             max_features="sqrt", n_jobs=-1, random_state=0)
EMBARGO_DEFAULT = 0.01
H_DEFAULT = 50                   # default label overlap (horizon, in bars), a
                                 # realistic max-holding; large enough vs the fold
                                 # size that overlap leakage is materially visible
H_GRID = [1, 5, 10, 25, 50, 100, 150]         # for inflation-vs-overlap sweep
EMBARGO_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]   # for inflation-vs-embargo sweep

INSTR = {
    "crypto": [f"{CRYPTO_CACHE}/{p}_1m.parquet" for p in
               ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT", "LTCUSDT", "LINKUSDT"]],
    "equity": [f"{ETF_DIR}/{p}.csv.gz" for p in ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]],
    "forex":  [f"{FX_CACHE}/{p}_fx1m.parquet" for p in
               ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURGBP"]],
}


# --------------------------------------------------------------------------- #
# Data loading -> information-driven bars (dogfoods Project 1)
# --------------------------------------------------------------------------- #
def load_bars(market, path):
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


# --------------------------------------------------------------------------- #
# Labels (the source of leakage), fixed-horizon, overlap span = H
# --------------------------------------------------------------------------- #
def fixed_horizon_label_py(close: np.ndarray, H: int) -> np.ndarray:
    """Pure-python reference: sign of the H-bar-ahead log return.
       label[t] = sign( log close[t+H] - log close[t] ),  uses ONLY future bars.
       Consecutive labels share H-1 of the same future bars => overlap span H."""
    n = close.shape[0]
    lp = np.log(close)
    y = np.full(n, np.nan)
    for t in range(n - H):
        y[t] = 1.0 if lp[t + H] > lp[t] else 0.0
    return y


@njit(cache=True)
def fixed_horizon_label_nb(close, H):
    n = close.shape[0]
    y = np.full(n, np.nan)
    for t in range(n - H):
        a = np.log(close[t + H]); b = np.log(close[t])
        y[t] = 1.0 if a > b else 0.0
    return y


# --------------------------------------------------------------------------- #
# Causal features  (everything uses only bars <= t)
# --------------------------------------------------------------------------- #
def causal_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].to_numpy(float)
    lp = np.log(close)
    r = np.diff(lp, prepend=lp[0])               # per-bar log return (causal)
    s = pd.DataFrame(index=bars.index)
    for w in (1, 3, 6, 12, 24):                  # trailing momentum (sum of past returns)
        s[f"mom{w}"] = pd.Series(r).rolling(w).sum().to_numpy()
    for w in (6, 24, 48):                         # trailing volatility
        s[f"vol{w}"] = pd.Series(r).rolling(w).std().to_numpy()
    # trailing volume z-score (causal)
    v = np.log1p(bars["volume"].to_numpy(float))
    s["vol_z"] = ((pd.Series(v) - pd.Series(v).rolling(48).mean())
                  / pd.Series(v).rolling(48).std()).to_numpy()
    # fractional-difference of log price (stationary, memory-preserving, causal)
    s["ffd"] = FD.ffd(lp, d=0.4, tau=1e-4)
    return s


# --------------------------------------------------------------------------- #
# Build a clean (X, y) design matrix for an instrument at overlap H
# --------------------------------------------------------------------------- #
def design_matrix(bars: pd.DataFrame, H: int):
    feats = causal_features(bars)
    y = fixed_horizon_label_nb(bars["close"].to_numpy(float), int(H))
    df = feats.copy(); df["y"] = y
    df = df.dropna()
    # keep time order (CV folds are contiguous in time)
    X = df.drop(columns="y").to_numpy(float)
    yv = df["y"].to_numpy(float)
    return X, yv


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _score(model, Xtr, ytr, Xte, yte):
    """Fit and return (accuracy, neg_log_loss, auc) on the test fold.
       Skips degenerate folds (single class in train or test)."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[: 1]
    yhat = (p >= 0.5).astype(float)
    acc = accuracy_score(yte, yhat)
    nll = -log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))     # higher = better
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = np.nan
    return acc, nll, auc


def cv_standard(X, y, n_splits=N_SPLITS):
    """Standard sklearn KFold (NON-shuffled, contiguous folds in time), leaky."""
    kf = KFold(n_splits=n_splits, shuffle=False)
    accs, nlls, aucs = [], [], []
    for tr, te in kf.split(X):
        m = RandomForestClassifier(**RF_KW)
        sc = _score(m, X[tr], y[tr], X[te], y[te])
        if sc:
            accs.append(sc[0]); nlls.append(sc[1]); aucs.append(sc[2])
    return _agg(accs, nlls, aucs)


def cv_purged(X, y, H, n_splits=N_SPLITS, embargo=EMBARGO_DEFAULT):
    """Purged k-fold + embargo (lib.overfit.purged_kfold_splits)."""
    accs, nlls, aucs = [], [], []
    for tr, te in OF.purged_kfold_splits(len(X), n_splits=n_splits,
                                         embargo_pct=embargo, label_span=H):
        m = RandomForestClassifier(**RF_KW)
        sc = _score(m, X[tr], y[tr], X[te], y[te])
        if sc:
            accs.append(sc[0]); nlls.append(sc[1]); aucs.append(sc[2])
    return _agg(accs, nlls, aucs)


def cv_cpcv(X, y, H, n_groups=6, k_test=2, embargo=EMBARGO_DEFAULT):
    """CPCV: score every combination of k_test held-out groups -> distribution."""
    accs, nlls, aucs = [], [], []
    for tr, te in OF.cpcv_splits(len(X), n_groups=n_groups, k_test=k_test,
                                 embargo_pct=embargo, label_span=H):
        m = RandomForestClassifier(**RF_KW)
        sc = _score(m, X[tr], y[tr], X[te], y[te])
        if sc:
            accs.append(sc[0]); nlls.append(sc[1]); aucs.append(sc[2])
    return dict(acc=np.array(accs), nll=np.array(nlls), auc=np.array(aucs))


def _agg(accs, nlls, aucs):
    f = lambda a: (float(np.mean(a)) if len(a) else np.nan)
    return dict(acc=f(accs), nll=f(nlls), auc=f(aucs),
                acc_all=np.array(accs), auc_all=np.array(aucs))


# --------------------------------------------------------------------------- #
# Per-instrument analysis at the default overlap H
# --------------------------------------------------------------------------- #
def analyse(market, path, H=H_DEFAULT, embargo=EMBARGO_DEFAULT, want_cpcv=False):
    name = os.path.basename(path).split("_")[0].replace(".csv.gz", "").replace(".parquet", "")
    bars = load_bars(market, path)
    if bars is None or len(bars) < 1500:
        return None
    X, y = design_matrix(bars, H)
    if len(X) < 1200 or len(np.unique(y)) < 2:
        return None
    std = cv_standard(X, y)
    pur = cv_purged(X, y, H, embargo=embargo)
    out = dict(market=market, instrument=name, n_obs=len(X), H=H,
               base_rate=float(y.mean()),
               kfold_acc=std["acc"], purged_acc=pur["acc"],
               kfold_auc=std["auc"], purged_auc=pur["auc"],
               kfold_nll=std["nll"], purged_nll=pur["nll"],
               infl_acc=std["acc"] - pur["acc"],
               infl_auc=std["auc"] - pur["auc"])
    if want_cpcv:
        out["_cpcv"] = cv_cpcv(X, y, H, embargo=embargo)
        out["_kfold_acc"] = std["acc"]; out["_purged_acc"] = pur["acc"]
    return out


# --------------------------------------------------------------------------- #
# Sweeps: inflation vs overlap H, inflation vs embargo
# --------------------------------------------------------------------------- #
def sweep_overlap(picks, embargo=EMBARGO_DEFAULT):
    rows = []
    for market, path in picks:
        bars = load_bars(market, path)
        if bars is None or len(bars) < 1500:
            continue
        name = os.path.basename(path).split("_")[0].replace(".csv.gz", "").replace(".parquet", "")
        for H in H_GRID:
            X, y = design_matrix(bars, H)
            if len(X) < 1200 or len(np.unique(y)) < 2:
                continue
            std = cv_standard(X, y); pur = cv_purged(X, y, H, embargo=embargo)
            rows.append(dict(market=market, instrument=name, H=H,
                             kfold_acc=std["acc"], purged_acc=pur["acc"],
                             infl_acc=std["acc"] - pur["acc"],
                             kfold_auc=std["auc"], purged_auc=pur["auc"],
                             infl_auc=std["auc"] - pur["auc"]))
            print(f"    [overlap] {market}:{name} H={H:3d} "
                  f"kfold_acc={std['acc']:.4f} purged_acc={pur['acc']:.4f} "
                  f"infl={std['acc']-pur['acc']:+.4f}")
    return pd.DataFrame(rows)


def sweep_embargo(picks, H=H_DEFAULT):
    rows = []
    for market, path in picks:
        bars = load_bars(market, path)
        if bars is None or len(bars) < 1500:
            continue
        name = os.path.basename(path).split("_")[0].replace(".csv.gz", "").replace(".parquet", "")
        X, y = design_matrix(bars, H)
        if len(X) < 1200 or len(np.unique(y)) < 2:
            continue
        std = cv_standard(X, y)
        for e in EMBARGO_GRID:
            pur = cv_purged(X, y, H, embargo=e)
            rows.append(dict(market=market, instrument=name, embargo=e, H=H,
                             kfold_acc=std["acc"], purged_acc=pur["acc"],
                             infl_acc=std["acc"] - pur["acc"],
                             kfold_auc=std["auc"], purged_auc=pur["auc"],
                             infl_auc=std["auc"] - pur["auc"]))
            print(f"    [embargo] {market}:{name} emb={e:.3f} "
                  f"purged_acc={pur['acc']:.4f} infl={std['acc']-pur['acc']:+.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figs(df, sw_over, sw_emb, cpcv_rep):
    d = f"{PROJ}/figures"

    # Fig 1, k-fold vs purged, by market (the leakage gap), accuracy AND AUC.
    # AUC is the sensitive detector (accuracy near a 50% base rate is coarse).
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, ksc, psc, lab in [(axes[0], "kfold_acc", "purged_acc", "accuracy"),
                              (axes[1], "kfold_auc", "purged_auc", "AUC")]:
        g = df.groupby("market")[[ksc, psc]].mean().reindex(["crypto", "equity", "forex"])
        x = np.arange(len(g)); w = 0.38
        ax.bar(x - w/2, g[ksc], w, label="standard k-fold (leaky)", color=ST.PALETTE["accent"])
        ax.bar(x + w/2, g[psc], w, label="purged k-fold + embargo (clean)", color=ST.PALETTE["dollar"])
        for i, m in enumerate(g.index):
            gap = (g.loc[m, ksc] - g.loc[m, psc]) * 100
            ax.text(i, max(g.loc[m, ksc], g.loc[m, psc]) + 0.004,
                    f"{gap:+.2f} pp", ha="center", fontsize=9,
                    color=ST.PALETTE["accent"] if gap >= 0 else "gray")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="coin-flip (0.50)")
        ax.set_xticks(x); ax.set_xticklabels(g.index)
        ax.set_ylabel(f"CV {lab}")
        ax.set_ylim(0.48, max(0.56, g.values.max() + 0.025))
        ax.set_title(f"{lab}: leaky vs clean")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Standard k-fold leaks vs purged+embargo  (fixed-horizon labels, overlap H={H_DEFAULT} bars)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{d}/fig1_kfold_vs_purged_by_market.png"); plt.close(fig)

    # Fig 2, inflation vs label overlap H, by market.
    # Two panels: accuracy (coarse 0/1 metric) and AUC (smoother, more sensitive
    # leakage detector). AUC is the cleaner signal because it uses the full
    # predicted probability instead of a thresholded label.
    cmap = {"crypto": "dollar", "equity": "volume", "forex": "tick"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True)
    for ax, col, lab in [(axes[0], "infl_acc", "accuracy"),
                         (axes[1], "infl_auc", "AUC")]:
        for m in ["crypto", "equity", "forex"]:
            sub = sw_over[sw_over.market == m].groupby("H")[col].mean()
            if len(sub):
                ax.plot(sub.index, sub.values * 100, marker="o",
                        label=m, color=ST.barcolor(cmap[m]))
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("label overlap H (bars, log scale)")
        ax.set_ylabel(f"CV {lab} inflation  (k-fold − purged)  [pp]")
        ax.set_title(f"{lab} inflation")
    axes[0].legend()
    fig.suptitle("Leakage grows with label overlap H  (more shared future bars => bigger inflation)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{d}/fig2_inflation_vs_overlap.png"); plt.close(fig)

    # Fig 3, residual inflation vs embargo size, by market (AUC = sensitive metric)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for m in ["crypto", "equity", "forex"]:
        sub = sw_emb[sw_emb.market == m].groupby("embargo")["infl_auc"].mean()
        if len(sub):
            ax.plot(sub.index * 100, sub.values * 100, marker="o",
                    label=m, color=ST.barcolor(cmap[m]))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("embargo size (% of sample)")
    ax.set_ylabel("residual AUC inflation  (k-fold − purged)  [pp]")
    ax.set_title(f"Embargo on top of full purging: little benefit, and it removes\n"
                 f"training data (purged k-fold already purges the H={H_DEFAULT} overlap)")
    ax.legend()
    fig.savefig(f"{d}/fig3_inflation_vs_embargo.png"); plt.close(fig)

    # Fig 4, CPCV OOS-score distribution for a representative instrument
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    paths = cpcv_rep["_cpcv"]["acc"]
    ax.hist(paths, bins=18, color=ST.PALETTE["dollar"], alpha=0.8,
            edgecolor="white", label=f"CPCV OOS paths (n={len(paths)})")
    ax.axvline(cpcv_rep["_purged_acc"], color="#333333", ls="--", lw=1.4,
               label=f"purged k-fold mean = {cpcv_rep['_purged_acc']:.3f}")
    ax.axvline(cpcv_rep["_kfold_acc"], color=ST.PALETTE["accent"], ls="--", lw=1.4,
               label=f"leaky single-split 'backtest' = {cpcv_rep['_kfold_acc']:.3f}")
    ax.axvline(paths.mean(), color=ST.PALETTE["tick"], lw=1.4,
               label=f"CPCV mean = {paths.mean():.3f}")
    ax.set_xlabel("OOS accuracy of a CPCV path")
    ax.set_ylabel("count")
    ax.set_title(f"CPCV: a backtest is ONE draw from a distribution of OOS paths\n"
                 f"({cpcv_rep['market']}:{cpcv_rep['instrument']}, overlap H={H_DEFAULT})")
    ax.legend(fontsize=8.5)
    fig.savefig(f"{d}/fig4_cpcv_distribution.png"); plt.close(fig)
    print("figures written:", sorted(os.listdir(d)))


# --------------------------------------------------------------------------- #
# Verify the Numba label kernel is bit-identical to the python reference
# --------------------------------------------------------------------------- #
def verify_kernel():
    rng = np.random.default_rng(7)
    print("=== label-kernel bit-identical check (python ref vs numba) ===")
    worst = 0.0
    for trial in range(6):
        n = rng.integers(2000, 6000)
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100
        for H in (1, 5, 20, 80):
            a = fixed_horizon_label_py(close, H)
            b = fixed_horizon_label_nb(close, int(H))
            # compare on the finite support (tail H entries are NaN in both)
            ma, mb = np.isnan(a), np.isnan(b)
            assert np.array_equal(ma, mb), f"NaN mask mismatch H={H}"
            delta = np.max(np.abs(a[~ma] - b[~mb])) if (~ma).any() else 0.0
            worst = max(worst, float(delta))
    print(f"  trials done. max|Δ| = {worst:.3e}  -> {'BIT-IDENTICAL' if worst == 0 else 'MISMATCH'}")
    return worst


# --------------------------------------------------------------------------- #
# Profile a single-instrument smoke test
# --------------------------------------------------------------------------- #
def profile_smoke():
    import cProfile, pstats, io
    print("=== cProfile smoke test: one crypto instrument, full CV ===")
    path = INSTR["crypto"][0]
    bars = load_bars("crypto", path)
    X, y = design_matrix(bars, H_DEFAULT)
    print(f"  {os.path.basename(path)}  n_obs={len(X)} features={X.shape[1]}")
    pr = cProfile.Profile()
    pr.enable()
    cv_standard(X, y)
    cv_purged(X, y, H_DEFAULT)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer instruments (smoke)")
    args = ap.parse_args()

    if args.verify:
        verify_kernel(); return
    if args.profile:
        verify_kernel(); profile_smoke(); return

    t0 = time.time()
    instr = INSTR
    if args.quick:
        instr = {m: paths[:2] for m, paths in INSTR.items()}

    # ---- per-instrument at default H ----
    rows, cpcv_rep = [], None
    print("=== per-instrument: standard k-fold vs purged+embargo (H=%d) ===" % H_DEFAULT)
    for market, paths in instr.items():
        for p in paths:
            if not os.path.exists(p):
                print("  missing", p); continue
            want_cpcv = (cpcv_rep is None and market == "crypto")
            try:
                r = analyse(market, p, want_cpcv=want_cpcv)
            except Exception as e:
                print("  fail", p, type(e).__name__, e); continue
            if r is None:
                continue
            if want_cpcv and "_cpcv" in r:
                cpcv_rep = r
            print(f"  {market}:{r['instrument']:9s} n={r['n_obs']:6d} "
                  f"kfold_acc={r['kfold_acc']:.4f} purged_acc={r['purged_acc']:.4f} "
                  f"infl={r['infl_acc']:+.4f}  (auc infl {r['infl_auc']:+.4f})")
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    df = pd.DataFrame(rows)
    df.to_csv(f"{PROJ}/tables/cv_per_instrument.csv", index=False)

    summ = df.groupby("market").agg(
        instruments=("instrument", "nunique"),
        n_obs=("n_obs", "median"),
        kfold_acc=("kfold_acc", "mean"), purged_acc=("purged_acc", "mean"),
        infl_acc=("infl_acc", "mean"),
        kfold_auc=("kfold_auc", "mean"), purged_auc=("purged_auc", "mean"),
        infl_auc=("infl_auc", "mean")).reindex(["crypto", "equity", "forex"])
    summ.to_csv(f"{PROJ}/tables/cv_by_market.csv")

    # ---- save the CPCV OOS-path scores (the distribution behind Fig 4) ----
    if cpcv_rep is not None and "_cpcv" in cpcv_rep:
        c = cpcv_rep["_cpcv"]
        pd.DataFrame({"acc": c["acc"], "auc": c["auc"], "nll": c["nll"]}).to_csv(
            f"{PROJ}/tables/cpcv_paths_{cpcv_rep['instrument']}.csv", index=False)

    # ---- sweeps ----
    # The overlap sweep is the headline figure; use a broad instrument set per
    # market (single-H inflation is noisy, so we average over many instruments).
    # The embargo sweep is lighter (fewer instruments) to keep RF cost sane.
    if args.quick:
        picks_over = [(m, instr[m][0]) for m in ["crypto", "equity", "forex"] if instr.get(m)]
        picks_emb = picks_over
    else:
        picks_over = [(m, p) for m in ["crypto", "equity", "forex"] for p in instr[m][:4]]
        picks_emb = [(m, p) for m in ["crypto", "equity", "forex"] for p in instr[m][:3]]
    print("\n=== sweep: inflation vs label overlap H ===")
    sw_over = sweep_overlap(picks_over)
    sw_over.to_csv(f"{PROJ}/tables/cv_inflation_vs_overlap.csv", index=False)
    print("\n=== sweep: inflation vs embargo size ===")
    sw_emb = sweep_embargo(picks_emb)
    sw_emb.to_csv(f"{PROJ}/tables/cv_inflation_vs_embargo.csv", index=False)

    # ---- markdown summary ----
    def _piv(sw, val, by):
        return (sw.groupby(["market", by])[val].mean().mul(100)
                .unstack(by).reindex(["crypto", "equity", "forex"]))
    with open(f"{PROJ}/tables/cv_by_market.md", "w") as fh:
        fh.write("# Cross-validation leakage at scale, mean by market\n\n")
        fh.write(f"RandomForest ({N_TREES} trees, depth 5, leaf 50) on causal features, "
                 f"fixed-horizon labels with overlap span H={H_DEFAULT} bars, {N_SPLITS}-fold CV, "
                 f"embargo {EMBARGO_DEFAULT*100:.0f}%. `kfold_*` = standard sklearn KFold (leaky); "
                 f"`purged_*` = purged k-fold + embargo (clean); `infl_*` = kfold − purged "
                 f"(the leakage). AUC is the sensitive detector; accuracy near a 50% base rate "
                 f"is high-variance.\n\n")
        fh.write(summ.round(4).to_markdown())
        fh.write("\n\n## Inflation vs label overlap H, AUC inflation (pp), the clean signal\n\n")
        fh.write(_piv(sw_over, "infl_auc", "H").round(2).to_markdown())
        fh.write("\n\n## Inflation vs label overlap H, accuracy inflation (pp), noisy\n\n")
        fh.write(_piv(sw_over, "infl_acc", "H").round(2).to_markdown())
        fh.write("\n\n## Residual inflation vs embargo, AUC (pp). At fixed H=50 the overlap is\n"
                 "already fully purged, so added embargo mostly removes training data, not leakage.\n\n")
        fh.write(_piv(sw_emb, "infl_auc", "embargo").round(2).to_markdown())

    print("\n=== SUMMARY BY MARKET ===\n", summ.round(4).to_string())
    make_figs(df, sw_over, sw_emb, cpcv_rep)
    print(f"\nProject 04 done in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()

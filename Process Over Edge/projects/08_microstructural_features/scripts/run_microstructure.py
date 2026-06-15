#!/usr/bin/env python3
"""
Project 08, Microstructural Features (Lopez de Prado, AFML Ch.19).

THE CLAIM (LdP, Ch.19).  Microstructure estimators distil the trading process,
the effective bid-ask spread, the price impact of order flow, and the toxicity of
that flow, into causal features that a model can read at the close of each bar.
Ch.19 catalogues a "second generation" of these built from bar data alone:
sequential trade models (Roll, Corwin-Schultz), strategic models of price impact
(Kyle, Amihud, Hasbrouck), and volume-clock toxicity (VPIN). The promise is that
these features carry information about *future* price moves and volatility that a
plain return/momentum feature set misses.

THE EXPERIMENT.  On real information-driven bars across THREE markets
(crypto perps, US-equity ETFs, forex majors) we:
  1. build the Ch.19 estimators as a causal feature library (micro_features.py),
  2. honestly gate which estimators each market can support (side-volume tiers),
  3. test predictive content for (a) next-bar DIRECTION and (b) next-bar
     VOLATILITY with a regularised RandomForest, scored under PURGED k-fold CV,
  4. run a COSTED long/short trading test off the direction signal, and feed the
     per-fold OOS Sharpe trials into the Deflated Sharpe Ratio + PBO gate
     (lib/overfit.py), the headline is DSR, NOT raw Sharpe.

CAUSALITY.  Every feature at bar t uses only bars <= t (verified by construction
in micro_features.py). The only forward-looking objects are the labels
(next-bar return sign / next-bar realised vol), exactly what purged CV protects.

COSTS.  The trading test is costed per house rules: a round-trip cost is charged
on every change of position (crypto/forex maker-ish + slippage; equities ETF
spread+commission). A cost-free signal test would be worthless, so the headline
trading metric is net-of-cost.

PROFILE-THEN-NUMBA.  --profile shows the rolling microstructure estimators
(Roll / Corwin-Schultz / Kyle / Hasbrouck / Amihud / VPIN) are the only
non-sklearn hot loop. They are moved to @njit kernels in micro_features.py and
VERIFIED bit-identical against pure-python references (--verify, prints max|d|).
RandomForest.fit dominates the remaining wall time; per house rules we do not
Numba sklearn.

Outputs (idempotent):
  tables/micro_availability.csv         , which estimators per market
  tables/micro_predictive.csv           , per-instrument dir/vol AUC, costed Sharpe
  tables/micro_by_market.csv/.md        , market roll-up + DSR/PBO
  tables/micro_dsr_pbo.csv              , headline overfitting gate
  figures/fig1_availability_matrix.png
  figures/fig2_predictive_auc_by_market.png
  figures/fig3_costed_sharpe_dsr.png
  figures/fig4_pbo_distribution.png

Rerun:
  python3 run_microstructure.py --verify    # bit-identical kernel check
  python3 run_microstructure.py --profile   # cProfile one-instrument smoke
  python3 run_microstructure.py --smoke      # 1-core tiny-subset end-to-end
  python3 run_microstructure.py              # full multi-market run (run_full.sh)
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bars as B
import overfit as OF
import realism as RZ
import style as ST
import micro_features as MF

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTO_CACHE = cfg.CRYPTO_1M
FX_CACHE = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M

BARS_PER_DAY = 8                 # ~3-hourly information bars (matches Projects 0/1/4)
N_SPLITS = 6
EMBARGO = 0.01
LABEL_SPAN = 1                   # next-bar labels => overlap span 1 (purge minimal but correct)
RF_DIR = dict(n_estimators=200, max_depth=5, min_samples_leaf=50,
              max_features="sqrt", n_jobs=1, random_state=0)   # n_jobs set at call site
RF_VOL = dict(n_estimators=200, max_depth=5, min_samples_leaf=50,
              max_features="sqrt", n_jobs=1, random_state=0)
VOL_WIN = 12                     # trailing realised-vol window for the vol label scale

# Per-market round-trip cost (fraction of notional), charged on each position change.
# Crypto perp: 0.02% maker + 0.02% slip per side ~ 0.08% RT. Equity ETF: ~1bp spread
# + commission ~ 0.02% RT. Forex major: ~0.5bp spread ~ 0.01% RT. Deliberately on the
# realistic side so the costed test is not a fantasy.
RT_COST = {"crypto": 0.0008, "equity": 0.0002, "forex": 0.0001}

# Instrument universe (diverse, multi-instrument per market).
INSTR_FULL = {
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT"],
    "equity": ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"],
    "forex":  ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURGBP"],
}
INSTR_SMOKE = {
    "crypto": ["BTCUSDT", "ETHUSDT"],
    "equity": ["SPY", "XLK"],
    "forex":  ["EURUSD", "GBPUSD"],
}
MARKETS = ["crypto", "equity", "forex"]


# --------------------------------------------------------------------------- #
# Data loading -> information-driven bars (dogfoods lib.bars)
# --------------------------------------------------------------------------- #
def _path(market, name):
    if market == "crypto":
        return f"{CRYPTO_CACHE}/{name}USDT_1m.parquet" if not name.endswith("USDT") \
            else f"{CRYPTO_CACHE}/{name}_1m.parquet"
    if market == "equity":
        return f"{ETF_DIR}/{name}.csv.gz"
    return f"{FX_CACHE}/{name}_fx1m.parquet"


# Smoke slice (rows of the 1m base kept, most-recent). Sized so days*BARS_PER_DAY
# clears the 1500-bar analysis floor in every market: crypto/forex 1m are dense
# (~700-1000 rows/day), equity RTH is ~390 rows/day, so equity needs fewer rows.
SMOKE_ROWS = {"crypto": 400_000, "equity": 120_000, "forex": 400_000}


def load_bars(market, name, smoke=False):
    """Return one information-driven bar series with the full lib.bars schema.
    smoke=True clips the base series to the most recent slice to keep it tiny."""
    if market == "crypto":
        df = pd.read_parquet(_path(market, name))
        df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns="open_time")
        df = df[(df.close > 0) & (df.volume > 0) & (df["count"] > 0)]
        if smoke:
            df = df.iloc[-SMOKE_ROWS["crypto"]:]
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.matched_bars(df, days * BARS_PER_DAY)["dollar"]
    if market == "equity":
        df = B.load_base_equity_etf(_path(market, name), rth=True)
        if smoke:
            df = df.iloc[-SMOKE_ROWS["equity"]:]
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.matched_bars(df, days * 6)["dollar"]
    if market == "forex":
        df = pd.read_parquet(_path(market, name))
        # forex parquet has only OHLC + count; synthesise the schema lib.bars needs.
        df["volume"] = df["count"]; df["quote_volume"] = df["count"]
        df["taker_buy_volume"] = df["count"] / 2; df["taker_buy_quote_volume"] = df["count"] / 2
        df = df[(df.close > 0) & (df["count"] > 0)]
        if smoke:
            df = df.iloc[-SMOKE_ROWS["forex"]:]
        days = max(50, int((df.index[-1] - df.index[0]).days))
        return B.threshold_bars(df, "count", df["count"].sum() / (days * BARS_PER_DAY))


# --------------------------------------------------------------------------- #
# Labels, next-bar direction & next-bar volatility (the only forward objects)
# --------------------------------------------------------------------------- #
def make_labels(bars: pd.DataFrame):
    lp = np.log(bars["close"].to_numpy(float))
    r = np.diff(lp, prepend=lp[0])              # per-bar log return (causal)
    r_next = np.roll(r, -1); r_next[-1] = np.nan        # next-bar return (forward)
    y_dir = (r_next > 0).astype(float)
    y_dir[np.isnan(r_next)] = np.nan
    # next-bar realised vol: |r_{t+1}| relative to trailing vol scale (forward label)
    vol_now = pd.Series(np.abs(r)).rolling(VOL_WIN).mean().to_numpy()
    y_vol = (np.abs(r_next) > vol_now).astype(float)    # is next move bigger than usual
    y_vol[np.isnan(r_next) | np.isnan(vol_now)] = np.nan
    return r, r_next, y_dir, y_vol


def design(bars: pd.DataFrame, market: str, sym: str = ""):
    feats = MF.build_features(bars, market)
    cols = [c for c in MF.available_features(market) if c in feats.columns]
    feats = feats[cols]
    r, r_next, y_dir, y_vol = make_labels(bars)
    d = feats.copy()
    d["_r_next"] = r_next; d["_ydir"] = y_dir; d["_yvol"] = y_vol
    d["_close"] = bars["close"].to_numpy(float)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    X = d[cols].to_numpy(float)
    # REALISTIC per-bar per-side cost aligned to the surviving rows (EQUITY/FOREX);
    # crypto keeps the flat house default. Causal (time-of-day schedule ex-ante).
    cost_side = RZ.per_side_cost_fraction(market, sym, d.index,
                                          d["_close"].to_numpy(float),
                                          crypto_fallback=RT_COST.get(market, 0.0008))
    return (X, d["_ydir"].to_numpy(float), d["_yvol"].to_numpy(float),
            d["_r_next"].to_numpy(float), cols, cost_side)


# --------------------------------------------------------------------------- #
# Purged-CV evaluation: direction AUC, vol AUC, and a costed long/short Sharpe
# --------------------------------------------------------------------------- #
def evaluate(X, y_dir, y_vol, r_next, market, cost_side=None, n_jobs=1):
    """Purged k-fold. Returns OOS direction AUC, vol AUC, and per-fold OOS
    net-of-cost per-bar Sharpe trials (one per fold) for the DSR gate.

    `cost_side` is a per-ROW per-side cost array (REALISTIC for equity/forex; flat
    house default for crypto). A position change at a row is charged TWICE the
    per-side cost (cross the spread to flip), i.e. round-trip turnover."""
    rf_dir = dict(RF_DIR); rf_dir["n_jobs"] = n_jobs
    rf_vol = dict(RF_VOL); rf_vol["n_jobs"] = n_jobs
    if cost_side is None:
        cost_side = np.full(len(X), RT_COST[market], float)
    else:
        cost_side = np.asarray(cost_side, float)
    dir_aucs, vol_aucs, fold_sharpes = [], [], []
    oos_pnl_all = []                            # concat OOS net returns for headline SR
    for tr, te in OF.purged_kfold_splits(len(X), n_splits=N_SPLITS,
                                         embargo_pct=EMBARGO, label_span=LABEL_SPAN):
        if len(np.unique(y_dir[tr])) < 2 or len(np.unique(y_dir[te])) < 2:
            continue
        # --- direction model + costed trade ---
        m = RandomForestClassifier(**rf_dir).fit(X[tr], y_dir[tr])
        p = m.predict_proba(X[te])[: 1]
        try:
            dir_aucs.append(roc_auc_score(y_dir[te], p))
        except Exception:
            pass
        pos = np.where(p >= 0.5, 1.0, -1.0)            # long/short on the signal
        gross = pos * r_next[te]                       # next-bar return realised
        turns = np.abs(np.diff(pos, prepend=0.0))      # position changes (flip=2.0)
        # per-side cost x |Δpos|: a flip (turns=2) pays one full round trip.
        net = gross - turns * cost_side[te]
        oos_pnl_all.append(net)
        sd = net.std(ddof=1)
        if sd > 0:
            fold_sharpes.append(net.mean() / sd)
        # --- volatility model ---
        if len(np.unique(y_vol[tr])) >= 2 and len(np.unique(y_vol[te])) >= 2:
            mv = RandomForestClassifier(**rf_vol).fit(X[tr], y_vol[tr])
            pv = mv.predict_proba(X[te])[: 1]
            try:
                vol_aucs.append(roc_auc_score(y_vol[te], pv))
            except Exception:
                pass
    net_all = np.concatenate(oos_pnl_all) if oos_pnl_all else np.array([])
    return dict(
        dir_auc=float(np.mean(dir_aucs)) if dir_aucs else np.nan,
        vol_auc=float(np.mean(vol_aucs)) if vol_aucs else np.nan,
        net_sharpe=OF.sharpe(net_all) if len(net_all) else np.nan,
        net_mean_bp=float(net_all.mean() * 1e4) if len(net_all) else np.nan,
        fold_sharpes=np.array(fold_sharpes),
        net_all=net_all)


def analyse(market, name, smoke=False, n_jobs=1):
    bars = load_bars(market, name, smoke=smoke)
    if bars is None or len(bars) < 1500:
        return None
    X, y_dir, y_vol, r_next, cols, cost_side = design(bars, market, name)
    if len(X) < 1200 or len(np.unique(y_dir)) < 2:
        return None
    ev = evaluate(X, y_dir, y_vol, r_next, market, cost_side=cost_side, n_jobs=n_jobs)
    return dict(market=market, instrument=name, n_obs=len(X), n_feat=len(cols),
                features=";".join(cols), dir_base=float(y_dir.mean()),
                **{k: v for k, v in ev.items()
                   if k in ("dir_auc", "vol_auc", "net_sharpe", "net_mean_bp")},
                _fold_sharpes=ev["fold_sharpes"], _net=ev["net_all"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figs(df, dsr_rows):
    d = f"{PROJ}/figures"
    cmap = {"crypto": "dollar", "equity": "volume", "forex": "tick"}

    # Fig 1, availability matrix (estimator x market)
    est = ["roll_spread", "corwin_schultz", "amihud", "tick_sign", "tick_flow",
           "kyle", "hasbrouck", "vpin", "ofi"]
    grid = np.zeros((len(est), len(MARKETS)))
    notes = {("kyle", "forex"): "tick", ("vpin", "equity"): "BVC",
             ("ofi", "equity"): "BVC", ("kyle", "equity"): "BVC",
             ("hasbrouck", "equity"): "BVC"}
    for j, mk in enumerate(MARKETS):
        avail = set(MF.available_features(mk))
        for i, e in enumerate(est):
            grid[i, j] = 1.0 if e in avail else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(MARKETS))); ax.set_xticklabels(MARKETS)
    ax.set_yticks(range(len(est))); ax.set_yticklabels(est)
    for i in range(len(est)):
        for j, mk in enumerate(MARKETS):
            if grid[i, j] > 0:
                tag = notes.get((est[i], mk), "true")
                ax.text(j, i, tag, ha="center", va="center", fontsize=8,
                        color="#11442b")
            else:
                ax.text(j, i, ", ", ha="center", va="center", color="#999")
    ax.set_title("Estimator availability by market\n(true = real side volume; BVC = bulk-volume proxy; tick = price-only)")
    fig.tight_layout(); fig.savefig(f"{d}/fig1_availability_matrix.png"); plt.close(fig)

    # Fig 2, predictive AUC by market (direction & vol)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, col, lab in [(axes[0], "dir_auc", "next-bar direction"),
                         (axes[1], "vol_auc", "next-bar volatility")]:
        g = df.groupby("market")[col].agg(["mean", "std"]).reindex(MARKETS)
        x = np.arange(len(g))
        ax.bar(x, g["mean"], 0.6, yerr=g["std"].fillna(0),
               color=[ST.barcolor(cmap[m]) for m in g.index], capsize=3)
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="coin-flip (0.50)")
        ax.set_xticks(x); ax.set_xticklabels(g.index)
        ax.set_ylabel("OOS AUC (purged k-fold)")
        ax.set_ylim(0.46, max(0.56, np.nanmax(g["mean"].values) + 0.03))
        ax.set_title(f"AUC: {lab}"); ax.legend(fontsize=8.5)
    fig.suptitle("Predictive content of microstructure features (purged-CV OOS AUC)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{d}/fig2_predictive_auc_by_market.png"); plt.close(fig)

    # Fig 3, costed net Sharpe per instrument + DSR benchmark
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    df2 = df.sort_values("net_sharpe")
    x = np.arange(len(df2))
    ax.bar(x, df2["net_sharpe"], color=[ST.barcolor(cmap[m]) for m in df2["market"]])
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df2["instrument"], rotation=60, fontsize=7.5)
    ax.set_ylabel("net-of-cost per-bar Sharpe (OOS)")
    ax.set_title("Costed long/short on the direction signal, net OOS Sharpe per instrument")
    fig.tight_layout(); fig.savefig(f"{d}/fig3_costed_sharpe_dsr.png"); plt.close(fig)

    # Fig 4, PBO / DSR gate summary
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    if dsr_rows:
        dd = pd.DataFrame(dsr_rows)
        x = np.arange(len(dd))
        ax.bar(x - 0.2, dd["dsr"], 0.4, label="DSR (selected best)", color=ST.PALETTE["dollar"])
        ax.bar(x + 0.2, dd["pbo"], 0.4, label="PBO (CSCV)", color=ST.PALETTE["accent"])
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(dd["scope"])
        ax.set_ylabel("probability"); ax.set_ylim(0, 1)
        ax.set_title("Overfitting gate: DSR (want high) vs PBO (want low)")
        ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{d}/fig4_pbo_distribution.png"); plt.close(fig)
    print("figures written:", sorted(os.listdir(d)))


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_smoke():
    import cProfile, pstats, io
    print("=== cProfile smoke: one crypto instrument, feature build + one CV fold ===")
    bars = load_bars("crypto", "BTCUSDT", smoke=True)
    print(f"  BTCUSDT smoke bars: {len(bars)}")
    pr = cProfile.Profile(); pr.enable()
    X, y_dir, y_vol, r_next, cols, cost_side = design(bars, "crypto", "BTCUSDT")
    # one fold of RF to show sklearn share
    tr, te = next(OF.purged_kfold_splits(len(X), n_splits=N_SPLITS,
                                         embargo_pct=EMBARGO, label_span=LABEL_SPAN))
    RandomForestClassifier(**{**RF_DIR, "n_jobs": 1}).fit(X[tr], y_dir[tr])
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    print(s.getvalue())

    # micro-only profile (the part we Numba): python ref vs numba on real data
    close = bars["close"].to_numpy(float)
    lp = np.log(close); dP = np.diff(lp, prepend=lp[0])
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
    sx = np.sign(MF.tick_rule(close)) * bars["volume"].to_numpy(float)
    MF._roll_spread_nb(np.ascontiguousarray(dP), MF.SPREAD_WIN)   # warm jit
    MF._ols_slope_nb(np.ascontiguousarray(dP), np.ascontiguousarray(sx), MF.LAMBDA_WIN)
    n_rep = 5
    t0 = time.time()
    for _ in range(n_rep):
        MF._roll_spread_py(dP, MF.SPREAD_WIN)
        MF._ols_slope_py(dP, sx, MF.LAMBDA_WIN)
        MF._roll_nanmean_py(np.abs(dP), MF.VPIN_WIN)
    t_py = (time.time() - t0) / n_rep
    t0 = time.time()
    for _ in range(n_rep):
        MF._roll_spread_nb(np.ascontiguousarray(dP), MF.SPREAD_WIN)
        MF._ols_slope_nb(np.ascontiguousarray(dP), np.ascontiguousarray(sx), MF.LAMBDA_WIN)
        MF._roll_nanmean_nb(np.ascontiguousarray(np.abs(dP)), MF.VPIN_WIN)
    t_nb = (time.time() - t0) / n_rep
    print(f"\n  rolling micro estimators (roll_spread + ols_slope + nanmean) on n={len(close)} bars:")
    print(f"    pure-python ref : {t_py*1e3:8.2f} ms")
    print(f"    numba kernels   : {t_nb*1e3:8.2f} ms")
    print(f"    speedup         : {t_py/max(t_nb,1e-9):8.1f}x")


# --------------------------------------------------------------------------- #
def write_availability_table():
    rows = []
    for mk in MARKETS:
        for e in MF.available_features(mk):
            tier = MF.MARKET_TIER[mk]
            kind = ("true" if tier == "side" and e in ("kyle", "hasbrouck", "vpin", "ofi")
                    else "BVC" if tier == "bvc" and e in ("kyle", "hasbrouck", "vpin", "ofi")
                    else "tick" if (mk == "forex" and e == "kyle")
                    else "price-only")
            rows.append(dict(market=mk, estimator=e, basis=kind))
    pd.DataFrame(rows).to_csv(f"{PROJ}/tables/micro_availability.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="bit-identical kernel check")
    ap.add_argument("--profile", action="store_true", help="cProfile + numba speedup")
    ap.add_argument("--smoke", action="store_true", help="1-core tiny-subset end-to-end")
    ap.add_argument("--jobs", type=int, default=-1, help="RF n_jobs (smoke forces 1)")
    args = ap.parse_args()

    if args.verify:
        w = MF.verify_kernels()
        print(f"=== kernel bit-identical check ===\n  max|delta| = {w:.3e}  "
              f"-> {'BIT-IDENTICAL' if w == 0 else 'MISMATCH'}")
        return
    if args.profile:
        w = MF.verify_kernels()
        print(f"kernel verify max|delta|={w:.3e} ({'BIT-IDENTICAL' if w==0 else 'MISMATCH'})")
        profile_smoke(); return

    t0 = time.time()
    smoke = args.smoke
    n_jobs = 1 if smoke else args.jobs
    instr = INSTR_SMOKE if smoke else INSTR_FULL
    write_availability_table()

    rows = []
    print(f"=== per-instrument microstructure predictive test "
          f"({'SMOKE' if smoke else 'FULL'}, n_jobs={n_jobs}) ===")
    for mk in MARKETS:
        for name in instr[mk]:
            p = _path(mk, name)
            if not os.path.exists(p):
                print("  missing", p); continue
            try:
                r = analyse(mk, name, smoke=smoke, n_jobs=n_jobs)
            except Exception as e:
                print("  fail", mk, name, type(e).__name__, e); continue
            if r is None:
                print(f"  skip {mk}:{name} (too few bars)"); continue
            print(f"  {mk}:{name:9s} n={r['n_obs']:6d} feat={r['n_feat']} "
                  f"dirAUC={r['dir_auc']:.4f} volAUC={r['vol_auc']:.4f} "
                  f"netSR={r['net_sharpe']:+.4f} netbp={r['net_mean_bp']:+.3f}")
            rows.append(r)

    if not rows:
        print("no instruments produced results"); return
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(f"{PROJ}/tables/micro_predictive.csv", index=False)

    # ---- market roll-up ----
    summ = df.groupby("market").agg(
        instruments=("instrument", "nunique"), n_obs=("n_obs", "median"),
        n_feat=("n_feat", "max"),
        dir_auc=("dir_auc", "mean"), vol_auc=("vol_auc", "mean"),
        net_sharpe=("net_sharpe", "mean"), net_mean_bp=("net_mean_bp", "mean")).reindex(MARKETS)
    summ.to_csv(f"{PROJ}/tables/micro_by_market.csv")
    with open(f"{PROJ}/tables/micro_by_market.md", "w") as fh:
        fh.write("# Project 08, microstructure features: per-market summary\n\n")
        fh.write(summ.round(4).to_markdown())
        fh.write("\n")

    # ---- DSR / PBO overfitting gate (the headline) ----
    # Trials = the costed per-bar Sharpe of every (instrument x fold). DSR deflates
    # the best of those trials by the number+dispersion of trials. PBO via CSCV on
    # the stacked OOS net-return matrix (instruments aligned by truncation).
    dsr_rows = []
    for scope, sub in [("all", rows)] + [(mk, [r for r in rows if r["market"] == mk]) for mk in MARKETS]:
        trials = np.concatenate([r["_fold_sharpes"] for r in sub if len(r["_fold_sharpes"])]) \
            if sub else np.array([])
        if len(trials) < 2:
            continue
        best = float(np.nanmax(trials))
        # use the OOS net-return series of the best instrument for skew/kurt + n_obs
        best_r = max(sub, key=lambda r: (np.nanmax(r["_fold_sharpes"])
                                         if len(r["_fold_sharpes"]) else -1e9))
        net = best_r["_net"]
        dres = OF.deflated_sharpe_ratio(
            best, len(net), float(ss.skew(net)), float(ss.kurtosis(net, fisher=False)), trials)
        # PBO: stack per-instrument OOS net returns, truncate to common length
        nets = [r["_net"] for r in sub if len(r["_net"]) > 500]
        pbo = np.nan
        if len(nets) >= 4:
            L = min(len(x) for x in nets)
            M = np.column_stack([x[:L] for x in nets])
            try:
                pbo = OF.pbo_cscv(M, n_splits=10)["pbo"]
            except Exception:
                pbo = np.nan
        dsr_rows.append(dict(scope=scope, n_trials=int(dres["n_trials"]),
                             best_sharpe=best, sr0=dres["sr0"], dsr=dres["dsr"],
                             pbo=pbo))
        print(f"  [gate] scope={scope:7s} n_trials={dres['n_trials']:3d} "
              f"best_SR={best:+.3f} SR0={dres['sr0']:.3f} DSR={dres['dsr']:.3f} "
              f"PBO={pbo:.3f}")
    pd.DataFrame(dsr_rows).to_csv(f"{PROJ}/tables/micro_dsr_pbo.csv", index=False)

    make_figs(df, dsr_rows)
    print(f"\nDONE in {time.time()-t0:.1f}s. tables+figures under {PROJ}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
stationary_vs_raw_flow.py: completeness item for Project 08 (Microstructural
Features): a head-to-head on how the order-flow inputs are PRESENTED to the model.

THE QUESTION.  Project 08 (and Ch.19) treats microstructure estimators as causal
features. A separate strand of the literature (Kolm, Turiel, Westray) argues that
for order-flow-driven prediction the STATIONARITY of the input matters more than
the model: a non-stationary level series fed raw into a learner degrades out of
sample, whereas the same information presented as a stationary transform predicts.
This script tests exactly that, holding EVERYTHING else fixed.

THE EXPERIMENT.  On crypto perps (the only tier with TRUE buyer/seller volume) we
build, from the study's own information-driven bars, a single order-flow LEVEL
series: the cumulative signed (buyer minus seller) dollar flow, a running net
order-flow imbalance. We then feed the SAME ML task / SAME purged CV the flow in
two presentations:

  (a) RAW LEVEL: the cumulative-flow level (and its one-bar lag): a
      non-stationary, trending series.
  (b) STATIONARY: order-flow imbalance (OFI, the per-bar signed split, already
      stationary) PLUS fractional differentiation (FFD) of the same cumulative
      level at the MINIMAL d that passes ADF while retaining level memory
      (lib/fracdiff.py).

Same labels (next-bar direction), same RandomForest, same purged k-fold CV, same
realistic per-side cost on the costed long/short sleeve. We report the OOS AUC and
net-of-cost Sharpe gap, raw vs stationary, per instrument and pooled, plus the d
used, and we deflate the best costed Sharpe with the DSR so the comparison is
honest about luck.

CAUSALITY.  The flow level uses only bars <= t (cumulative sum is causal). OFI at
bar t uses only bar t. FFD is a causal fixed-width backward filter (lib/fracdiff).
The minimal-d ADF search is run on the TRAINING span only inside each fold and the
chosen d is applied to the held-out span (no look-ahead from the test fold into d
selection). All features are lagged one bar before they meet a next-bar label.

COSTS.  The long/short sleeve charges the study's realistic per-side crypto cost on
every position change (a flip pays a full round trip), identical for both variants.

Outputs (idempotent):
  tables/stationary_vs_raw_flow.csv / .md   : per instrument + pooled, raw vs stat
  figures/fig_stationary_vs_raw.png         : the gap

Run:
  python3 stationary_vs_raw_flow.py --smoke   # 2 instruments, tiny slice
  python3 stationary_vs_raw_flow.py           # default instrument subset
"""
from __future__ import annotations
import sys, os, argparse, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
sys.path.insert(0, cfg.LIB)
sys.path.insert(0, HERE)
import bars as B
import overfit as OF
import realism as RZ
import style as ST
import fracdiff as FD
import micro_features as MF
import run_microstructure as RUN          # reuse load_bars / labels / CV constants

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(HERE)
MARKET = "crypto"                          # only tier with true buyer/seller volume

# Subset of the study's crypto universe (bounded for RAM; one bar series at a time).
INSTR = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
INSTR_SMOKE = ["BTCUSDT", "ETHUSDT"]

# Minimal-d FFD search grid (coarse enough to be cheap, fine enough to be honest).
D_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 4)
FFD_TAU = 1e-4                             # weight-truncation -> modest fixed window

N_SPLITS = RUN.N_SPLITS
EMBARGO = RUN.EMBARGO
LABEL_SPAN = RUN.LABEL_SPAN
RF = dict(RUN.RF_DIR)                      # identical model to the study's direction RF


# --------------------------------------------------------------------------- #
# Order-flow level + the two presentations
# --------------------------------------------------------------------------- #
def flow_series(bars: pd.DataFrame):
    """Causal order-flow objects from the study's own bars (true signed dollar).

    Returns (level, ofi):
      level : cumulative signed (buyer minus seller) dollar flow, normalised by a
              causal trailing scale so it is comparable across instruments. A
              trending, NON-STATIONARY level series. level[t] uses bars <= t.
      ofi   : per-bar signed order-flow imbalance in [-1, 1] (stationary by
              construction). ofi[t] uses only bar t.
    """
    buy_d = bars["buy_dollar"].to_numpy(float)
    sell_d = bars["sell_dollar"].to_numpy(float)
    net = buy_d - sell_d                                  # signed dollar flow per bar
    tot = buy_d + sell_d
    ofi = np.where(tot > 0, net / tot, 0.0)               # stationary per-bar split
    cum = np.cumsum(net)                                  # causal cumulative level
    # causal trailing scale (expanding std of per-bar net) so the level is unit-free
    scale = pd.Series(net).expanding(min_periods=20).std().to_numpy()
    scale = np.where((scale > 0) & np.isfinite(scale), scale, np.nan)
    level = cum / scale
    return level.astype(float), ofi.astype(float)


def _train_min_d(level_tr: np.ndarray) -> float:
    """Minimal d on D_GRID whose FFD of the (training-only) level passes ADF.
    Falls back to the largest grid d if none passes. Training-span only -> the d
    choice never sees the held-out fold."""
    x = level_tr[np.isfinite(level_tr)]
    if len(x) < 200:
        return float(D_GRID[-1])
    res = FD.min_d_search(x, d_grid=D_GRID, tau=FFD_TAU, signif="5%")
    d = res["d_star"]
    return float(d) if np.isfinite(d) else float(D_GRID[-1])


# --------------------------------------------------------------------------- #
# Build the aligned design: labels, raw block, stationary block, costs
# --------------------------------------------------------------------------- #
def build(bars: pd.DataFrame, sym: str):
    r, r_next, y_dir, _ = RUN.make_labels(bars)
    level, ofi = flow_series(bars)
    close = bars["close"].to_numpy(float)
    idx = bars.index
    # Lag every feature one bar so nothing at t uses information dated t for a
    # label that resolves over (t, t+1]. (Labels are already next-bar.)
    level_lag = np.roll(level, 1); level_lag[0] = np.nan
    ofi_lag = np.roll(ofi, 1); ofi_lag[0] = np.nan
    level_lag2 = np.roll(level, 2); level_lag2[:2] = np.nan      # raw: level + its lag
    return dict(idx=idx, close=close, r_next=r_next, y_dir=y_dir,
                level=level, level_lag=level_lag, level_lag2=level_lag2,
                ofi_lag=ofi_lag, sym=sym)


def _finite_mask(*arrays):
    m = np.ones(len(arrays[0]), bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def evaluate_variant(D, variant: str, cost_side: np.ndarray):
    """Purged k-fold for one presentation of the flow.

    variant='raw'        -> features = [level_lag, level_lag2]   (non-stationary)
    variant='stationary' -> features = [ofi_lag, FFD(level)_lag] (stationary)
    The FFD d is searched on each fold's TRAINING span only, then applied to the
    whole level series and read at the held-out rows (causal filter, no leak)."""
    y = D["y_dir"]; r_next = D["r_next"]; cost = cost_side
    aucs, fold_sharpes, oos = [], [], []
    d_used = []
    n = len(y)
    for tr, te in OF.purged_kfold_splits(n, n_splits=N_SPLITS,
                                         embargo_pct=EMBARGO, label_span=LABEL_SPAN):
        if variant == "raw":
            cols = [D["level_lag"], D["level_lag2"]]
        else:
            d = _train_min_d(D["level"][tr])
            d_used.append(d)
            ffd_full = FD.ffd(D["level"], d, tau=FFD_TAU)       # causal filter, full len
            ffd_lag = np.roll(ffd_full, 1); ffd_lag[0] = np.nan  # lag like the rest
            cols = [D["ofi_lag"], ffd_lag]
        X = np.column_stack(cols)
        m = _finite_mask(*cols, y, r_next, cost)
        tr_m = tr[m[tr]]; te_m = te[m[te]]
        if len(tr_m) < 100 or len(te_m) < 30:
            continue
        if len(np.unique(y[tr_m])) < 2 or len(np.unique(y[te_m])) < 2:
            continue
        mdl = RandomForestClassifier(**{**RF, "n_jobs": 1}).fit(X[tr_m], y[tr_m])
        p = mdl.predict_proba(X[te_m])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te_m], p))
        except Exception:
            pass
        pos = np.where(p >= 0.5, 1.0, -1.0)
        gross = pos * r_next[te_m]
        turns = np.abs(np.diff(pos, prepend=0.0))
        net = gross - turns * cost[te_m]
        oos.append(net)
        sd = net.std(ddof=1)
        if sd > 0:
            fold_sharpes.append(net.mean() / sd)
    net_all = np.concatenate(oos) if oos else np.array([])
    return dict(
        auc=float(np.mean(aucs)) if aucs else np.nan,
        net_sharpe=OF.sharpe(net_all) if len(net_all) else np.nan,
        net_mean_bp=float(net_all.mean() * 1e4) if len(net_all) else np.nan,
        d_used=float(np.median(d_used)) if d_used else np.nan,
        fold_sharpes=np.array(fold_sharpes),
        net_all=net_all,
    )


def analyse(sym: str, smoke=False):
    bars = RUN.load_bars(MARKET, sym, smoke=smoke)
    if bars is None or len(bars) < 1500:
        return None
    D = build(bars, sym)
    cost_side = RZ.per_side_cost_fraction(
        MARKET, sym, D["idx"], D["close"],
        crypto_fallback=RUN.RT_COST.get(MARKET, 0.0008))
    raw = evaluate_variant(D, "raw", cost_side)
    stat = evaluate_variant(D, "stationary", cost_side)
    return dict(instrument=sym, n_obs=int(np.isfinite(D["y_dir"]).sum()),
                d_used=stat["d_used"],
                auc_raw=raw["auc"], auc_stat=stat["auc"],
                auc_gap=stat["auc"] - raw["auc"],
                sr_raw=raw["net_sharpe"], sr_stat=stat["net_sharpe"],
                sr_gap=stat["net_sharpe"] - raw["net_sharpe"],
                bp_raw=raw["net_mean_bp"], bp_stat=stat["net_mean_bp"],
                _raw=raw, _stat=stat)


# --------------------------------------------------------------------------- #
def make_fig(df: pd.DataFrame, pooled: dict):
    d = f"{PROJ}/figures"
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    x = np.arange(len(df)); w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, df["auc_raw"], w, label="raw level",
           color=ST.barcolor("tick"))
    ax.bar(x + w / 2, df["auc_stat"], w, label="stationary (OFI + FFD)",
           color=ST.barcolor("dollar"))
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="coin-flip (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(df["instrument"], rotation=45, fontsize=8)
    ax.set_ylabel("OOS direction AUC (purged k-fold)")
    lo = min(0.48, float(np.nanmin([df["auc_raw"].min(), df["auc_stat"].min()])) - 0.005)
    hi = max(0.53, float(np.nanmax([df["auc_raw"].max(), df["auc_stat"].max()])) + 0.005)
    ax.set_ylim(lo, hi)
    ax.set_title("Predictive AUC: raw level vs stationary flow")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.bar(x - w / 2, df["sr_raw"], w, label="raw level", color=ST.barcolor("tick"))
    ax.bar(x + w / 2, df["sr_stat"], w, label="stationary (OFI + FFD)",
           color=ST.barcolor("dollar"))
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df["instrument"], rotation=45, fontsize=8)
    ax.set_ylabel("net-of-cost per-bar Sharpe (OOS)")
    ax.set_title("Costed long/short Sharpe: raw vs stationary")
    ax.legend(fontsize=8.5)

    fig.suptitle("Order-flow inputs: stationarity vs raw level "
                 f"(crypto, pooled AUC gap {pooled['auc_gap']:+.4f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{d}/fig_stationary_vs_raw.png"
    fig.savefig(out); plt.close(fig)
    print("figure written:", out)


def _pooled_dsr(rows, key):
    """Deflated Sharpe of the best costed fold-Sharpe across instruments for one
    variant ('_raw' or '_stat')."""
    trials = np.concatenate([r[key]["fold_sharpes"] for r in rows
                             if len(r[key]["fold_sharpes"])]) \
        if rows else np.array([])
    if len(trials) < 2:
        return dict(best=np.nan, sr0=np.nan, dsr=np.nan, n_trials=len(trials))
    best = float(np.nanmax(trials))
    best_r = max(rows, key=lambda r: (np.nanmax(r[key]["fold_sharpes"])
                                      if len(r[key]["fold_sharpes"]) else -1e9))
    net = best_r[key]["net_all"]
    dres = OF.deflated_sharpe_ratio(best, len(net), float(ss.skew(net)),
                                    float(ss.kurtosis(net, fisher=False)), trials)
    return dict(best=best, sr0=dres["sr0"], dsr=dres["dsr"],
                n_trials=int(dres["n_trials"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    instr = INSTR_SMOKE if args.smoke else INSTR

    rows = []
    print(f"=== stationary-vs-raw order-flow ({'SMOKE' if args.smoke else 'FULL'}) ===")
    for sym in instr:
        p = RUN._path(MARKET, sym)
        if not os.path.exists(p):
            print("  missing", p); continue
        try:
            r = analyse(sym, smoke=args.smoke)
        except Exception as e:
            print("  fail", sym, type(e).__name__, e); continue
        if r is None:
            print(f"  skip {sym} (too few bars)"); continue
        print(f"  {sym:9s} n={r['n_obs']:6d} d*={r['d_used']:.2f}  "
              f"AUC raw={r['auc_raw']:.4f} stat={r['auc_stat']:.4f} "
              f"gap={r['auc_gap']:+.4f} | SR raw={r['sr_raw']:+.4f} "
              f"stat={r['sr_stat']:+.4f} gap={r['sr_gap']:+.4f}")
        rows.append(r)

    if not rows:
        print("no instruments produced results"); return

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])

    pooled = dict(
        instrument="POOLED",
        n_obs=int(df["n_obs"].sum()),
        d_used=float(df["d_used"].median()),
        auc_raw=float(df["auc_raw"].mean()), auc_stat=float(df["auc_stat"].mean()),
        auc_gap=float((df["auc_stat"] - df["auc_raw"]).mean()),
        sr_raw=float(df["sr_raw"].mean()), sr_stat=float(df["sr_stat"].mean()),
        sr_gap=float((df["sr_stat"] - df["sr_raw"]).mean()),
        bp_raw=float(df["bp_raw"].mean()), bp_stat=float(df["bp_stat"].mean()),
    )

    dsr_raw = _pooled_dsr(rows, "_raw")
    dsr_stat = _pooled_dsr(rows, "_stat")
    pooled["dsr_raw"] = dsr_raw["dsr"]
    pooled["dsr_stat"] = dsr_stat["dsr"]

    out = pd.concat([df, pd.DataFrame([pooled])], ignore_index=True)
    cols = ["instrument", "n_obs", "d_used", "auc_raw", "auc_stat", "auc_gap",
            "sr_raw", "sr_stat", "sr_gap", "bp_raw", "bp_stat",
            "dsr_raw", "dsr_stat"]
    out = out.reindex(columns=cols)
    out.to_csv(f"{PROJ}/tables/stationary_vs_raw_flow.csv", index=False)

    with open(f"{PROJ}/tables/stationary_vs_raw_flow.md", "w") as fh:
        fh.write("# Project 08: order-flow inputs, stationary transform vs raw level\n\n")
        fh.write("Same ML task / purged CV / cost. Crypto only (true buyer-seller "
                 "volume). RAW = cumulative signed-flow level (+ its lag). "
                 "STATIONARY = order-flow imbalance + FFD of that level at the "
                 "minimal d (per-fold, training-span ADF) that retains memory.\n\n")
        fh.write(out.round(4).to_markdown(index=False))
        fh.write("\n\n")
        fh.write(f"- Pooled DSR (best costed fold-Sharpe, deflated): "
                 f"raw {dsr_raw['dsr']:.3f} (best SR {dsr_raw['best']:+.3f}, "
                 f"E[max] {dsr_raw['sr0']:.3f}, n_trials {dsr_raw['n_trials']}); "
                 f"stationary {dsr_stat['dsr']:.3f} (best SR {dsr_stat['best']:+.3f}, "
                 f"E[max] {dsr_stat['sr0']:.3f}, n_trials {dsr_stat['n_trials']}).\n")

    make_fig(df, pooled)
    print(f"\n[pooled] AUC raw={pooled['auc_raw']:.4f} stat={pooled['auc_stat']:.4f} "
          f"gap={pooled['auc_gap']:+.4f} | SR raw={pooled['sr_raw']:+.4f} "
          f"stat={pooled['sr_stat']:+.4f} gap={pooled['sr_gap']:+.4f}")
    print(f"[pooled] DSR raw={pooled['dsr_raw']:.3f} stat={pooled['dsr_stat']:.3f}")
    print(f"DONE in {time.time()-t0:.1f}s. tables+figure under {PROJ}")


if __name__ == "__main__":
    main()

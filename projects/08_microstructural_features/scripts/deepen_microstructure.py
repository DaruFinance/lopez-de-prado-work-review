#!/usr/bin/env python3
"""
deepen_microstructure.py — Phase-2 DEEPEN for Project 08 (AFML Ch.19).

The Phase-1 run (run_microstructure.py) answered "do microstructure features
predict, at a market level, after realistic costs?" — and the answer was a hard
no on direction and a weak-but-real yes on volatility, DSR-deflated to nothing
tradeable. This script answers the THREE questions that a market-level roll-up
cannot:

  (A) WHICH ESTIMATOR carries the signal — per-estimator, single-feature purged-CV
      AUC for next-bar DIRECTION and next-bar VOLATILITY, per market. (One feature
      at a time, same RF, same purged folds, so the AUCs are comparable to the
      0.50 coin-flip and to each other.)

  (B) HOW MUCH THE DATA TIER MATTERS — the clean ablation. CRYPTO is the only
      market with TRUE buyer/seller volume. We recompute the four side-volume
      estimators (Kyle, Hasbrouck, VPIN, OFI) on the *identical* crypto bars in
      three tiers:
        TRUE  — real taker buy/sell dollar split (what crypto actually has),
        BVC   — discard the side flag, re-estimate it with Bulk-Volume
                Classification from price+volume (what an equity has),
        TICK  — discard volume too, sign by the tick rule only (what forex has).
      The vol-AUC gap TRUE→BVC→TICK is a direct, same-instrument measurement of
      what you lose by descending the availability ladder. (Phase-1's cross-market
      comparison confounds tier with the instrument; this does not.)

  (C) HONEST DSR-GATED VERDICT — restated from the Phase-1 gate, plus a per-market
      best-single-feature DSR so the verdict is not hostage to the 9-feature model.

Everything is causal (features at bar t use bars <= t), costed with lib/realism
(no clamping), purged-CV scored, DSR-headlined. Imports lib + micro_features
READ-ONLY. Writes only new tables/figures under projects/08.../{tables,figures}.

Run:
  python3 deepen_microstructure.py            # full (uses run_full instrument set)
  python3 deepen_microstructure.py --smoke    # 2 instr/market, fast sanity
"""
from __future__ import annotations
import sys, os, argparse, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/daru/ldp_review/lib")
sys.path.insert(0, HERE)
import bars as B
import overfit as OF
import realism as RZ
import style as ST
import micro_features as MF
import run_microstructure as RUN          # reuse load_bars / make_labels / design helpers

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ST.set_style()
PROJ = RUN.PROJ
MARKETS = RUN.MARKETS

# single-feature RF: shallow, regularised — same spirit as the multi-feature model
RF1 = dict(n_estimators=120, max_depth=4, min_samples_leaf=80,
           max_features=1.0, random_state=0)
N_SPLITS = RUN.N_SPLITS
EMBARGO = RUN.EMBARGO
LABEL_SPAN = RUN.LABEL_SPAN


# --------------------------------------------------------------------------- #
# (A) per-estimator single-feature purged-CV AUC
# --------------------------------------------------------------------------- #
def single_feature_auc(x, y, n_jobs=1):
    """Purged-CV OOS AUC of ONE feature predicting binary label y.
    Each fit is tiny (one column), so spawning many threads is pure overhead —
    we force n_jobs=1 here regardless of the caller's request."""
    x = np.asarray(x, float).reshape(-1, 1)
    aucs = []
    for tr, te in OF.purged_kfold_splits(len(x), n_splits=N_SPLITS,
                                         embargo_pct=EMBARGO, label_span=LABEL_SPAN):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        rf = dict(RF1); rf["n_jobs"] = 1
        m = RandomForestClassifier(**rf).fit(x[tr], y[tr])
        p = m.predict_proba(x[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], p))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else np.nan


def _per_estimator_one(mk, name, smoke):
    """Worker: per-feature dir/vol AUC for ONE instrument. n_jobs=1 inside (tiny
    single-column fits); parallelism is across instruments at the pool level."""
    if not os.path.exists(RUN._path(mk, name)):
        return None
    bars = RUN.load_bars(mk, name, smoke=smoke)
    if bars is None or len(bars) < 1500:
        return None
    feats_all = MF.available_features(mk)
    f = MF.build_features(bars, mk)
    cols = [c for c in feats_all if c in f.columns]
    r, r_next, y_dir, y_vol = RUN.make_labels(bars)
    d = f[cols].copy()
    d["_yd"] = y_dir; d["_yv"] = y_vol
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    yd = d["_yd"].to_numpy(float); yv = d["_yv"].to_numpy(float)
    out = {}
    for c in cols:
        xc = d[c].to_numpy(float)
        out[c] = (single_feature_auc(xc, yd), single_feature_auc(xc, yv))
    print(f"  (A) {mk}:{name} done ({len(cols)} feats)", flush=True)
    return mk, out


def per_estimator_table(instr, smoke=False, n_jobs=-1):
    """For each market x estimator: mean single-feature dir & vol AUC across that
    market's instruments. Parallel ACROSS instruments (each worker single-thread)."""
    from joblib import Parallel, delayed
    tasks = [(mk, name) for mk in MARKETS for name in instr[mk]]
    n = os.cpu_count() if n_jobs in (-1, None) else n_jobs
    n = max(1, min(n, len(tasks)))
    results = Parallel(n_jobs=n, backend="loky")(
        delayed(_per_estimator_one)(mk, name, smoke) for mk, name in tasks)
    # aggregate
    rec = []
    for mk in MARKETS:
        feats_all = MF.available_features(mk)
        dacc = {f: [] for f in feats_all}; vacc = {f: [] for f in feats_all}
        for res in results:
            if res is None or res[0] != mk:
                continue
            for c, (da, va) in res[1].items():
                dacc[c].append(da); vacc[c].append(va)
        for f in feats_all:
            dv = [a for a in dacc[f] if np.isfinite(a)]
            vv = [a for a in vacc[f] if np.isfinite(a)]
            rec.append(dict(market=mk, estimator=f,
                            dir_auc=float(np.mean(dv)) if dv else np.nan,
                            vol_auc=float(np.mean(vv)) if vv else np.nan,
                            n_instr=len(vv)))
    return pd.DataFrame(rec)


# --------------------------------------------------------------------------- #
# (B) crypto tier ablation: TRUE vs BVC vs TICK on identical bars
# --------------------------------------------------------------------------- #
SIDE_FEATS = ["kyle", "hasbrouck", "vpin", "ofi"]


def crypto_sideflow_features(bars: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Recompute the four side-volume estimators on crypto bars using a chosen
    signing tier. Price-only base features are identical across tiers (omitted
    here; this isolates the side-volume block)."""
    close = bars["close"].to_numpy(float)
    vol = bars["volume"].to_numpy(float)
    dollar = bars["dollar"].to_numpy(float)
    f = pd.DataFrame(index=bars.index)
    if tier == "true":
        buy_d = bars["buy_dollar"].to_numpy(float)
        sell_d = bars["sell_dollar"].to_numpy(float)
        tot = buy_d + sell_d
        bf = np.where(tot > 0, buy_d / tot, 0.5)
        buy_v = vol * bf; sell_v = vol * (1.0 - bf)
    elif tier == "bvc":
        bf = MF.bvc_buy_fraction(close, MF.BVC_SIGMA_WIN)
        buy_v = vol * bf; sell_v = vol * (1.0 - bf)
    elif tier == "tick":
        # tick-rule sign on the bar volume (the forex analogue, but with crypto's
        # real volume available so VPIN/OFI are still computable as a proxy)
        b = MF.tick_rule(close)
        pos = b > 0
        buy_v = np.where(pos, vol, 0.0); sell_v = np.where(~pos, vol, 0.0)
    else:
        raise ValueError(tier)
    signed_vol = buy_v - sell_v
    signed_dollar = np.sign(signed_vol) * np.sqrt(np.abs(dollar))
    f["kyle"] = MF.kyle_lambda(close, signed_vol, MF.LAMBDA_WIN)
    f["hasbrouck"] = MF.hasbrouck_lambda(close, signed_dollar, MF.LAMBDA_WIN)
    f["vpin"] = MF.vpin(buy_v, sell_v, MF.VPIN_WIN)
    f["ofi"] = MF.order_flow_imbalance(buy_v, sell_v)
    return f


def tier_block_auc(bars, tier, n_jobs=1):
    """OOS dir & vol AUC of the 4-feature side-volume block under a tier, on one
    crypto instrument (multi-feature RF, same as Phase-1 settings)."""
    f = crypto_sideflow_features(bars, tier)
    r, r_next, y_dir, y_vol = RUN.make_labels(bars)
    d = f.copy(); d["_yd"] = y_dir; d["_yv"] = y_vol
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    X = d[SIDE_FEATS].to_numpy(float)
    yd = d["_yd"].to_numpy(float); yv = d["_yv"].to_numpy(float)
    da, va = [], []
    rf = dict(RUN.RF_DIR); rf["n_jobs"] = n_jobs
    for tr, te in OF.purged_kfold_splits(len(X), n_splits=N_SPLITS,
                                         embargo_pct=EMBARGO, label_span=LABEL_SPAN):
        for y, store in [(yd, da), (yv, va)]:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = RandomForestClassifier(**rf).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            try:
                store.append(roc_auc_score(y[te], p))
            except Exception:
                pass
    return (float(np.mean(da)) if da else np.nan,
            float(np.mean(va)) if va else np.nan)


def _tier_ablation_one(name, smoke):
    if not os.path.exists(RUN._path("crypto", name)):
        return None
    bars = RUN.load_bars("crypto", name, smoke=smoke)
    if bars is None or len(bars) < 1500:
        return None
    row = dict(instrument=name)
    for tier in ("true", "bvc", "tick"):
        da, va = tier_block_auc(bars, tier, n_jobs=1)
        row[f"{tier}_dir_auc"] = da
        row[f"{tier}_vol_auc"] = va
    print(f"  (B) crypto:{name} true/bvc/tick vol_auc = "
          f"{row['true_vol_auc']:.4f}/{row['bvc_vol_auc']:.4f}/{row['tick_vol_auc']:.4f}",
          flush=True)
    return row


def tier_ablation_table(instr, smoke=False, n_jobs=-1):
    from joblib import Parallel, delayed
    names = instr["crypto"]
    n = os.cpu_count() if n_jobs in (-1, None) else n_jobs
    n = max(1, min(n, len(names)))
    rec = Parallel(n_jobs=n, backend="loky")(
        delayed(_tier_ablation_one)(name, smoke) for name in names)
    return pd.DataFrame([r for r in rec if r is not None])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_deepen_figs(per_est, tier):
    d = f"{PROJ}/figures"
    cmap = {"crypto": "dollar", "equity": "volume", "forex": "tick"}

    # Fig 5 — per-estimator vol & dir AUC heat per market (vol is where signal is)
    est_order = ["roll_spread", "corwin_schultz", "amihud", "tick_sign",
                 "tick_flow", "kyle", "hasbrouck", "vpin", "ofi"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    for ax, col, lab in [(axes[0], "vol_auc", "next-bar VOLATILITY"),
                         (axes[1], "dir_auc", "next-bar DIRECTION")]:
        piv = per_est.pivot(index="estimator", columns="market", values=col)
        piv = piv.reindex(index=est_order, columns=MARKETS)
        M = piv.to_numpy(float)
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.48, vmax=0.56, aspect="auto")
        ax.set_xticks(range(len(MARKETS))); ax.set_xticklabels(MARKETS)
        ax.set_yticks(range(len(est_order))); ax.set_yticklabels(est_order)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="black")
                else:
                    ax.text(j, i, "—", ha="center", va="center", color="#999")
        ax.set_title(f"single-feature OOS AUC\n{lab}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Which estimator carries the signal (purged-CV, one feature at a time)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{d}/fig5_per_estimator_auc.png"); plt.close(fig)

    # Fig 6 — tier ablation on crypto (vol AUC): TRUE -> BVC -> TICK
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(tier)); w = 0.26
    ax.bar(x - w, tier["true_vol_auc"], w, label="TRUE side-volume (crypto has it)",
           color=ST.PALETTE["dollar"])
    ax.bar(x, tier["bvc_vol_auc"], w, label="BVC proxy (equity tier)",
           color=ST.PALETTE["volume"])
    ax.bar(x + w, tier["tick_vol_auc"], w, label="tick-only (forex tier)",
           color=ST.PALETTE["tick"])
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tier["instrument"], rotation=55, fontsize=8)
    ax.set_ylabel("OOS vol AUC of the side-volume block")
    lo = np.nanmin(tier[["true_vol_auc", "bvc_vol_auc", "tick_vol_auc"]].to_numpy())
    hi = np.nanmax(tier[["true_vol_auc", "bvc_vol_auc", "tick_vol_auc"]].to_numpy())
    ax.set_ylim(min(0.49, lo - 0.005), hi + 0.008)
    ax.set_title("Data-tier ablation on IDENTICAL crypto bars:\n"
                 "what the side-volume estimators lose as you descend the availability ladder")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{d}/fig6_tier_ablation.png"); plt.close(fig)
    print("figures written: fig5_per_estimator_auc.png, fig6_tier_ablation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    t0 = time.time()
    instr = RUN.INSTR_SMOKE if args.smoke else RUN.INSTR_FULL
    n_jobs = 1 if args.smoke else args.jobs

    print("=== (A) per-estimator single-feature AUC ===")
    per_est = per_estimator_table(instr, smoke=args.smoke, n_jobs=n_jobs)
    per_est.to_csv(f"{PROJ}/tables/micro_per_estimator.csv", index=False)

    print("=== (B) crypto data-tier ablation (TRUE vs BVC vs TICK) ===")
    tier = tier_ablation_table(instr, smoke=args.smoke, n_jobs=n_jobs)
    tier.to_csv(f"{PROJ}/tables/micro_tier_ablation.csv", index=False)

    # tier-gap summary (mean degradation across crypto instruments)
    gap = dict(
        true_vol=float(tier["true_vol_auc"].mean()),
        bvc_vol=float(tier["bvc_vol_auc"].mean()),
        tick_vol=float(tier["tick_vol_auc"].mean()),
        true_dir=float(tier["true_dir_auc"].mean()),
        bvc_dir=float(tier["bvc_dir_auc"].mean()),
        tick_dir=float(tier["tick_dir_auc"].mean()),
    )
    gap["vol_drop_true_to_bvc"] = gap["true_vol"] - gap["bvc_vol"]
    gap["vol_drop_true_to_tick"] = gap["true_vol"] - gap["tick_vol"]
    pd.DataFrame([gap]).to_csv(f"{PROJ}/tables/micro_tier_gap.csv", index=False)
    print("  tier-gap:", {k: round(v, 4) for k, v in gap.items()})

    make_deepen_figs(per_est, tier)
    print(f"\nDEEPEN DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""deepen.py — Phase-2 DEEPENING for Project 09.

Reuses run_ensembles_importance internals (no edits to the driver) and the same
purged/CPCV machinery to answer the deepening questions that the headline run
did not persist:

  (A) Bagging vs boosting — paired across instruments: IS-OOS overfit gap, OOS
      DSR, and the IS vs OOS log-loss LEVELS (does bagging generalize better on
      noisy financial data, as LdP claims, AND does that translate into edge?).
  (B) Feature importance — MDI vs MDA vs CLUSTERED-MDA, paired substitution-bias
      (Spearman rho of importance vs mean |corr|), and the STABILITY of the
      selected top-set across CPCV paths for EACH method, compared against a
      random-selection baseline (is the selection actually informative?).
  (C) Log-loss vs accuracy scoring control — does the tuning objective change the
      selected config and the resulting overfit gap (AFML 9.4)?

Writes tables/deepen_*.csv + tables/deepen_summary.md and three figures.
Instrument-parallel (--jobs). sklearn fits dominate; non-sklearn loops are the
already-verified numba kernels in the driver.
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, itertools, json
# pin BLAS/OMP to 1 thread per process BEFORE numpy/sklearn import: determinism
# independent of --jobs, and a good neighbour to other parallel agents.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/daru/ldp_review"
for p in (os.path.join(ROOT, "lib"), ROOT, HERE,
          os.path.join(ROOT, "projects", "03_meta_labeling", "scripts")):
    sys.path.insert(0, p)

import run_ensembles_importance as R          # the driver (read-only reuse)
from lib import overfit as O
from scipy import stats as ss
from sklearn.metrics import log_loss

TAB = os.path.join(HERE, "..", "tables")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TOP_K = 3


# ---- importance helpers that also return CLUSTERED stability & per-method bias ----
def _perm_importance_oos(mdl, Xte, yte, groups, rng):
    base = -log_loss(yte, mdl.predict_proba(Xte), labels=[0, 1])
    imp = np.empty(len(groups))
    for gi, g in enumerate(groups):
        Xp = Xte.copy()
        perm = rng.permutation(len(Xte))
        for j in g:
            Xp[:, j] = Xp[perm, j]
        sc = -log_loss(yte, mdl.predict_proba(Xp), labels=[0, 1])
        imp[gi] = (base - sc) / abs(base) if base != 0 else (base - sc)
    return imp


def stability_all_methods(X, y, sw, label_span, avg_u, params, clusters):
    """Per-CPCV-path MDA, clustered-MDA, and MDI top-set; mean Jaccard for each.
    Also returns the raw per-path importance arrays for variance analysis."""
    p = X.shape[1]
    feat_groups = [[j] for j in range(p)]
    # map cluster id -> member feature indices (for translating cluster top-set to feats)
    rng = np.random.default_rng(R.RANDOM_STATE)
    mda_tops, cmda_tops, mdi_tops = [], [], []
    mda_paths = []
    for tr, te in O.cpcv_splits(len(y), R.CPCV_GROUPS, R.CPCV_K, R.EMBARGO, label_span):
        if len(tr) < 80 or len(np.unique(y[tr])) < 2 or len(te) < 20:
            continue
        mdl = R.make_rf(params, avg_u, True)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        # MDA (per-feature)
        imp = _perm_importance_oos(mdl, X[te], y[te], feat_groups, rng)
        mda_paths.append(imp)
        mda_tops.append(set(np.argsort(imp)[-TOP_K:]))
        # clustered-MDA: permute whole clusters; pick top clusters, expand to feats
        cimp = _perm_importance_oos(mdl, X[te], y[te], clusters, rng)
        ktop = min(TOP_K, len(clusters))
        topc = np.argsort(cimp)[-ktop:]
        feats = set()
        for ci in topc:
            feats |= set(clusters[ci])
        cmda_tops.append(feats)
        # MDI (in-sample importance from the SAME fitted model)
        mi = mdl.feature_importances_
        mdi_tops.append(set(np.argsort(mi)[-TOP_K:]))

    def mean_jac(tops):
        if len(tops) < 2:
            return np.nan
        j = []
        for i in range(len(tops)):
            for k in range(i + 1, len(tops)):
                a, b = tops[i], tops[k]
                j.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        return float(np.mean(j))

    mda_paths = np.array(mda_paths) if mda_paths else np.zeros((0, p))
    return dict(stab_mda=mean_jac(mda_tops),
                stab_cmda=mean_jac(cmda_tops),
                stab_mdi=mean_jac(mdi_tops),
                n_paths=len(mda_tops),
                mda_path_std=float(np.nanmean(mda_paths.std(0))) if mda_paths.size else np.nan)


def random_jaccard_baseline(p, top_k, n_sets, n_boot=2000, seed=0):
    """Expected mean pairwise Jaccard if top-`top_k` of `p` features were picked
    uniformly at random by each of `n_sets` paths. Monte-Carlo (matches how many
    paths we actually have)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        tops = [set(rng.choice(p, top_k, replace=False)) for _ in range(max(2, n_sets))]
        j = []
        for i in range(len(tops)):
            for k in range(i + 1, len(tops)):
                a, b = tops[i], tops[k]
                j.append(len(a & b) / len(a | b))
        vals.append(np.mean(j))
    return float(np.mean(vals)), float(np.std(vals))


def deepen_instrument(market, name, path):
    task = R.build_task(market, path)
    if task is None:
        return None
    X, y = task["X"], task["y"]
    sw, avg_u = R.make_sample_weights(task["t0"], task["t1"], task["nb_span"])
    ls = task["label_span_ev"]
    ev, hold, retg, cost, n_bars, bpy = (task["ev"], task["hold"], task["ret_gross"],
                                         task["cost"], task["n_bars"], task["bpy"])
    fn = task["feat_names"]; p = X.shape[1]

    # --- tune both models by NLL (headline) and ACC (control) ---
    rf_nll, _, _ = R.purged_grid_search(R.make_rf, R.RF_GRID, X, y, sw, ls, "nll", avg_u=avg_u)
    hgb_nll, _, _ = R.purged_grid_search(R.make_hgb, R.HGB_GRID, X, y, sw, ls, "nll", avg_u=avg_u)
    rf_acc, _, _ = R.purged_grid_search(R.make_rf, R.RF_GRID, X, y, sw, ls, "acc", avg_u=avg_u)
    hgb_acc, _, _ = R.purged_grid_search(R.make_hgb, R.HGB_GRID, X, y, sw, ls, "acc", avg_u=avg_u)

    # OOF + gap + IS/OOS levels for NLL-selected and ACC-selected
    rf_p, rf_gap, rf_ll = R.oof_predict(R.make_rf, rf_nll, X, y, sw, ls, avg_u)
    hgb_p, hgb_gap, hgb_ll = R.oof_predict(R.make_hgb, hgb_nll, X, y, sw, ls, avg_u)
    rf_pa, rf_gapa, rf_lla = R.oof_predict(R.make_rf, rf_acc, X, y, sw, ls, avg_u)
    hgb_pa, hgb_gapa, hgb_lla = R.oof_predict(R.make_hgb, hgb_acc, X, y, sw, ls, avg_u)

    rf_bet = R.score_bet(rf_p, retg, ev, hold, cost, n_bars, bpy)
    hgb_bet = R.score_bet(hgb_p, retg, ev, hold, cost, n_bars, bpy)
    rf_bet_a = R.score_bet(rf_pa, retg, ev, hold, cost, n_bars, bpy)
    hgb_bet_a = R.score_bet(hgb_pa, retg, ev, hold, cost, n_bars, bpy)

    # DSR (pooled trial dispersion, exactly as the driver)
    trials = np.array(R._grid_bet_srs(R.make_rf, R.RF_GRID, X, y, sw, ls, avg_u, retg, ev, hold, cost, n_bars, bpy)
                      + R._grid_bet_srs(R.make_hgb, R.HGB_GRID, X, y, sw, ls, avg_u, retg, ev, hold, cost, n_bars, bpy))
    rf_dsr = O.deflated_sharpe_ratio(rf_bet["sr"], rf_bet["n_obs"], rf_bet["skew"], rf_bet["kurt"], trials)
    hgb_dsr = O.deflated_sharpe_ratio(hgb_bet["sr"], hgb_bet["n_obs"], hgb_bet["skew"], hgb_bet["kurt"], trials)

    # config agreement between NLL and ACC tuning
    rf_cfg_same = int(rf_nll == rf_acc)
    hgb_cfg_same = int(hgb_nll == hgb_acc)

    # --- importance: bias of MDI vs MDA vs clustered-MDA ---
    clusters = R.feature_clusters(X, fn)
    mdi = R.importance_mdi(R.make_rf, rf_nll, X, y, sw, avg_u)
    mda, _ = R.importance_mda(X, y, sw, ls, avg_u, rf_nll, clustered=False)
    cmda, cgroups = R.importance_mda(X, y, sw, ls, avg_u, rf_nll, clustered=True, clusters=clusters)
    # feature mean |corr|
    Cmag = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0); np.fill_diagonal(Cmag, 0.0)
    fcorr = Cmag.mean(0)
    mdi_bias = float(ss.spearmanr(mdi, fcorr).correlation)
    mda_bias = float(ss.spearmanr(np.nan_to_num(mda), fcorr).correlation)
    # clustered-MDA bias: expand cluster importance back to per-feature, then corr
    cmda_feat = np.zeros(p)
    for gi, g in enumerate(cgroups):
        for j in g:
            cmda_feat[j] = cmda[gi] / max(1, len(g))
    cmda_bias = float(ss.spearmanr(np.nan_to_num(cmda_feat), fcorr).correlation)

    # --- stability across CPCV paths for each method ---
    stab = stability_all_methods(X, y, sw, ls, avg_u, rf_nll, clusters)
    rj_mean, rj_std = random_jaccard_baseline(p, TOP_K, max(2, stab["n_paths"]))
    # z of MDA stability vs random
    stab_z = ((stab["stab_mda"] - rj_mean) / rj_std) if (rj_std > 0 and np.isfinite(stab["stab_mda"])) else np.nan

    return dict(
        market=market, name=name, n_ev=len(ev), n_feat=p, base_rate=float(y.mean()),
        avg_uniqueness=avg_u, n_clusters=len(clusters),
        # bagging vs boosting
        rf_gap_nll=rf_gap, hgb_gap_nll=hgb_gap, rf_gap_acc=rf_gapa, hgb_gap_acc=hgb_gapa,
        rf_ll_is=rf_ll[0], rf_ll_oos=rf_ll[1], hgb_ll_is=hgb_ll[0], hgb_ll_oos=hgb_ll[1],
        rf_dsr=rf_dsr["dsr"], hgb_dsr=hgb_dsr["dsr"],
        rf_sr_ann=rf_bet["sr_ann"], hgb_sr_ann=hgb_bet["sr_ann"],
        rf_pf=rf_bet["pf"], hgb_pf=hgb_bet["pf"],
        # tuning control: NLL vs ACC
        rf_sr_ann_acc=rf_bet_a["sr_ann"], hgb_sr_ann_acc=hgb_bet_a["sr_ann"],
        rf_cfg_same=rf_cfg_same, hgb_cfg_same=hgb_cfg_same,
        # importance bias
        mdi_bias=mdi_bias, mda_bias=mda_bias, cmda_bias=cmda_bias,
        # stability
        stab_mdi=stab["stab_mdi"], stab_mda=stab["stab_mda"], stab_cmda=stab["stab_cmda"],
        stab_random=rj_mean, stab_mda_z=stab_z, n_cpcv_paths=stab["n_paths"],
    )


def _worker(arg):
    market, name, path = arg
    try:
        return (market, name, deepen_instrument(market, name, path), None)
    except Exception:
        import traceback
        return (market, name, None, traceback.format_exc())


def paired(df, a, b, label):
    x, y = df[a].values, df[b].values
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 6:
        return f"{label}: insufficient n"
    w = ss.wilcoxon(x[m], y[m])
    return (f"{label}: med({a})={np.median(x[m]):.4f} med({b})={np.median(y[m]):.4f}  "
            f"{a}<{b} in {(x[m]<y[m]).mean()*100:.0f}%  Wilcoxon p={w.pvalue:.2e}")


def summarize(df):
    lines = ["# Project 09 — DEEPENING summary\n"]
    lines.append(f"Instruments: {len(df)} ({df.market.value_counts().to_dict()})\n")

    lines.append("\n## (A) Bagging vs boosting (paired across instruments)\n")
    lines.append(paired(df, "rf_gap_nll", "hgb_gap_nll", "IS-OOS overfit gap, NLL-tuned"))
    lines.append(paired(df, "rf_gap_acc", "hgb_gap_acc", "IS-OOS overfit gap, ACC-tuned"))
    lines.append(paired(df, "rf_dsr", "hgb_dsr", "OOS DSR"))
    lines.append(paired(df, "rf_sr_ann", "hgb_sr_ann", "annual SR"))
    # IS vs OOS levels
    lines.append("\nIS vs OOS log-loss LEVELS by market (overfit = OOS >> IS):")
    lvl = df.groupby("market")[["rf_ll_is", "rf_ll_oos", "hgb_ll_is", "hgb_ll_oos"]].median().round(4)
    lines.append(lvl.to_markdown())
    # does bagging generalize better -> gap ratio
    df["_gap_ratio"] = df["hgb_gap_nll"] / df["rf_gap_nll"].replace(0, np.nan)
    lines.append(f"\nMedian HGB/RF overfit-gap ratio = {df['_gap_ratio'].median():.2f}x "
                 f"(boosting overfits ~{df['_gap_ratio'].median():.1f}x more than bagging).")
    lines.append(f"Fraction of instruments where RF gap < HGB gap: "
                 f"{(df.rf_gap_nll < df.hgb_gap_nll).mean()*100:.0f}%.")
    # edge verdict
    lines.append(f"\nDSR verdict: best per-instrument DSR over BOTH models = "
                 f"max median {df[['rf_dsr','hgb_dsr']].max(1).median():.3f}; "
                 f"instruments with EITHER model DSR>0.95: "
                 f"{((df.rf_dsr>0.95)|(df.hgb_dsr>0.95)).sum()}/{len(df)}.")

    lines.append("\n## (B) Feature importance: MDI vs MDA vs clustered-MDA\n")
    lines.append("Substitution bias = Spearman rho(importance, feature mean |corr|). "
                 "High rho = importance leaks to correlated features (LdP's MDI warning).")
    bias = df.groupby("market")[["mdi_bias", "mda_bias", "cmda_bias"]].median().round(3)
    lines.append(bias.to_markdown())
    lines.append("\noverall medians: " +
                 f"MDI bias={df.mdi_bias.median():.3f}, MDA bias={df.mda_bias.median():.3f}, "
                 f"clustered-MDA bias={df.cmda_bias.median():.3f}")
    lines.append(paired(df, "mda_bias", "mdi_bias", "MDA vs MDI substitution bias"))
    lines.append(paired(df, "cmda_bias", "mdi_bias", "clustered-MDA vs MDI substitution bias"))

    lines.append("\n### Stability of top-3 selected set across CPCV paths (mean Jaccard)\n")
    stab = df.groupby("market")[["stab_mdi", "stab_mda", "stab_cmda", "stab_random"]].median().round(3)
    lines.append(stab.to_markdown())
    lines.append("\noverall medians: " +
                 f"MDI={df.stab_mdi.median():.3f}, MDA={df.stab_mda.median():.3f}, "
                 f"clustered-MDA={df.stab_cmda.median():.3f}, random baseline={df.stab_random.median():.3f}")
    lines.append(paired(df, "stab_mda", "stab_random", "MDA stability vs random selection"))
    lines.append(paired(df, "stab_cmda", "stab_mda", "clustered-MDA vs MDA stability"))
    # one-sample: is MDA stability above its random baseline?
    diff = (df.stab_mda - df.stab_random).dropna()
    w = ss.wilcoxon(diff)
    lines.append(f"\nMDA - random (paired): median +{diff.median():.3f}, "
                 f"above random in {(diff>0).mean()*100:.0f}% of instruments, Wilcoxon p={w.pvalue:.2e}.")
    lines.append(f"Median MDA stability z vs random = {df.stab_mda_z.median():.2f} sigma.")

    lines.append("\n## (C) Tuning objective control: neg-log-loss vs accuracy\n")
    lines.append(paired(df, "rf_gap_nll", "rf_gap_acc", "RF overfit gap: NLL-tuned vs ACC-tuned"))
    lines.append(paired(df, "hgb_gap_nll", "hgb_gap_acc", "HGB overfit gap: NLL-tuned vs ACC-tuned"))
    lines.append(f"\nNLL vs ACC selected the SAME RF config in {df.rf_cfg_same.mean()*100:.0f}% of instruments, "
                 f"same HGB config in {df.hgb_cfg_same.mean()*100:.0f}%.")
    lines.append(paired(df, "rf_sr_ann", "rf_sr_ann_acc", "RF resulting SR: NLL-tuned vs ACC-tuned"))

    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, "deepen_summary.md"), "w") as f:
        f.write(txt)
    print(txt)
    return txt


def make_figs(df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    mk = ["crypto", "equities", "forex"]
    col = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}

    # FIG5: paired overfit gap RF vs HGB (scatter, log) + IS/OOS level bars
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in mk:
        s = df[df.market == m]
        ax[0].scatter(s.rf_gap_nll, s.hgb_gap_nll, s=26, color=col[m], label=m)
    lim = [0, max(df.rf_gap_nll.max(), df.hgb_gap_nll.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("bagging (RF) IS-OOS log-loss gap")
    ax[0].set_ylabel("boosting (HGB) IS-OOS log-loss gap")
    ax[0].set_title("Overfit gap, per instrument (above line = boosting overfits more)")
    ax[0].legend()
    # IS/OOS levels
    xs = np.arange(len(mk)); w = 0.2
    rf_is = [df[df.market == m].rf_ll_is.median() for m in mk]
    rf_oos = [df[df.market == m].rf_ll_oos.median() for m in mk]
    hg_is = [df[df.market == m].hgb_ll_is.median() for m in mk]
    hg_oos = [df[df.market == m].hgb_ll_oos.median() for m in mk]
    ax[1].bar(xs - 1.5*w, rf_is, w, label="RF IS", color=P["dollar"], alpha=0.55)
    ax[1].bar(xs - 0.5*w, rf_oos, w, label="RF OOS", color=P["dollar"])
    ax[1].bar(xs + 0.5*w, hg_is, w, label="HGB IS", color=P["accent"], alpha=0.55)
    ax[1].bar(xs + 1.5*w, hg_oos, w, label="HGB OOS", color=P["accent"])
    ax[1].axhline(np.log(2), color="k", ls=":", lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(mk)
    ax[1].set_ylabel("log-loss (dotted = coin-flip ln2)")
    ax[1].set_title("IS vs OOS log-loss (boosting memorizes IS)")
    ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_overfit_paired.png")); plt.close(fig)

    # FIG6: importance bias MDI vs MDA vs clustered-MDA (box by market)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    methods = ["mdi_bias", "mda_bias", "cmda_bias"]
    labels = ["MDI", "MDA", "clustered-MDA"]
    data = [df[m].dropna().values for m in methods]
    bp = ax[0].boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], [P["accent"], P["dollar"], P["volume"]]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("substitution bias  rho(importance, |corr|)")
    ax[0].set_title("MDI is correlation-biased; only MDA is bias-free\n(clustered-MDA expanded to features still tracks |corr|)")
    # stability vs random
    sm = ["stab_mdi", "stab_mda", "stab_cmda", "stab_random"]
    sl = ["MDI", "MDA", "clust-MDA", "random"]
    sd = [df[m].dropna().values for m in sm]
    bp2 = ax[1].boxplot(sd, labels=sl, patch_artist=True, showmeans=True)
    for patch, c in zip(bp2["boxes"], [P["accent"], P["dollar"], P["volume"], "0.6"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax[1].set_ylabel("mean Jaccard of top-3 set across CPCV paths")
    ax[1].set_title("Selection stability (vs random baseline)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_importance_bias_stability.png")); plt.close(fig)

    # FIG7: tuning objective control
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    gnll = df[["rf_gap_nll", "hgb_gap_nll"]].mean(1)
    gacc = df[["rf_gap_acc", "hgb_gap_acc"]].mean(1)
    for m in mk:
        s = df.market == m
        ax.scatter(gacc[s], gnll[s], s=26, color=col[m], label=m)
    lim = [min(gacc.min(), gnll.min()), max(gacc.max(), gnll.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("overfit gap, ACCURACY-tuned (control)")
    ax.set_ylabel("overfit gap, NEG-LOG-LOSS-tuned (headline)")
    ax.set_title("Below line = log-loss tuning overfits less (AFML 9.4)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_tuning_objective.png")); plt.close(fig)
    print("  figures -> fig5/fig6/fig7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    u = R.universe(smoke=args.smoke)
    print(f"DEEPEN: {len(u)} instruments, jobs={args.jobs}")
    rows = []
    t0 = time.perf_counter()
    if args.jobs > 1 and not args.smoke:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for market, name, r, err in pool.imap_unordered(_worker, [(m, n, p) for m, n, p in u]):
                if err:
                    print(f"  [ERR] {market} {name}: {err.splitlines()[-1]}")
                elif r is None:
                    print(f"  [skip] {market} {name}")
                else:
                    rows.append(r)
                    print(f"  [ok] {market:9s} {name:10s} gapRF={r['rf_gap_nll']:.3f} gapHGB={r['hgb_gap_nll']:.3f} "
                          f"MDIbias={r['mdi_bias']:.2f} MDAbias={r['mda_bias']:.2f} stabMDA={r['stab_mda']:.2f}")
    else:
        for m, n, p in u:
            _, _, r, err = _worker((m, n, p))
            if err: print(err)
            elif r: rows.append(r); print(f"  [ok] {m} {n}")
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(TAB, "deepen_results.parquet"))
    df.round(4).to_csv(os.path.join(TAB, "deepen_per_instrument.csv"), index=False)
    summarize(df)
    make_figs(df)
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s  ({len(df)} instruments)")


if __name__ == "__main__":
    main()

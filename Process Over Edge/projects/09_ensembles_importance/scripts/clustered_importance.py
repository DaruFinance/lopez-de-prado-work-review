#!/usr/bin/env python3
"""
clustered_importance.py: CLUSTERED feature importance (Lopez de Prado's remedy
for the substitution effect), added to the ensembles + feature-importance study.

The parent study (run_ensembles_importance.py) already shows that flat MDI is
substitution-biased: correlated features split each other's importance credit, so
a genuinely-informative block of correlated features can each look weak in
isolation. AFML 8.5 / ML4AM 6 prescribe the fix: group correlated features into
clusters, then measure importance PER CLUSTER.

  - clustered-MDI = SUM of the flat per-feature MDI within each cluster.
  - clustered-MDA = permute the WHOLE cluster's columns at once under purged CV
                    (one OOS log-loss drop per cluster).

This script reuses the parent study's exact pipeline (same one instrument it uses
as its crypto representative, BTCUSDT; same triple-barrier labeled task, same RF
recipe, same purged CV, same data path) and adds ONLY the cluster-level analysis:

  (1) flat MDI / MDA (the parent's baseline) on one instrument,
  (2) clustered MDI / MDA via hierarchical (Ward) clustering on the 1-|corr|
      feature-distance matrix,
  (3) the payoff: importance CONCENTRATION (how much the substitution effect
      dilutes a correlated block before clustering) and rank STABILITY across
      purged folds (cluster-level ranking is steadier than per-feature ranking).

Real data, real model, purged CV, no synthetic, no look-ahead. lib/ is reused,
never modified.

Run:
  python3 scripts/clustered_importance.py            # BTCUSDT (study's crypto rep)
  python3 scripts/clustered_importance.py --light    # tiny RF (fast sanity)
"""
from __future__ import annotations
import os, sys, argparse, warnings, resource
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# Reuse the parent study's pipeline UNCHANGED (data path, task, model, CV, flat
# MDI/MDA, the clustered-MDA permutation kernel, the cluster builder).
import run_ensembles_importance as P
from lib import overfit as O

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# The instrument the parent study uses as its crypto representative.
INSTRUMENT = ("crypto", "BTCUSDT",
              os.path.join(P.CRYPTO_DIR, "BTCUSDT_1m.parquet"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ward_clusters(X, max_k=6):
    """Hierarchical (Ward) clustering on the 1-|corr| feature-distance matrix.
    Returns (clusters, labels, linkage_Z). Ward on the squareform distance, cut
    at max_k clusters. Mirrors the parent's |corr| distance but with Ward linkage
    (the ML4AM-style variance-minimizing merge) instead of average linkage."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    C = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    D = np.sqrt(np.clip(0.5 * (1.0 - C), 0.0, 1.0))   # 0 when |corr|=1
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                                # enforce symmetry
    k = min(max_k, X.shape[1] - 1)
    Z = linkage(squareform(D, checks=False), method="ward")
    lab = fcluster(Z, t=k, criterion="maxclust")
    clusters = [list(np.where(lab == c)[0]) for c in np.unique(lab)]
    return clusters, lab, Z


def gini(x):
    """Gini concentration of a non-negative importance vector (0 = perfectly
    even split, higher = more concentrated). Negative entries clipped to 0."""
    x = np.sort(np.clip(np.asarray(x, float), 0.0, None))
    n = len(x)
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)


def per_fold_flat_importance(X, y, sw, label_span, avg_u, params, n_splits):
    """Per-purged-fold FLAT importance: MDI (in-fold impurity of the fold's RF
    fit) and MDA (permutation drop on the fold's OOS block). Returns two arrays
    of shape (n_folds, n_features). Used for the cross-fold stability comparison
    against the cluster-level ranking."""
    from sklearn.metrics import log_loss
    p = X.shape[1]
    mdi_rows, mda_rows = [], []
    rng = np.random.default_rng(P.RANDOM_STATE)
    for tr, te in O.purged_kfold_splits(len(y), n_splits, P.EMBARGO, label_span):
        if len(tr) < 60 or len(np.unique(y[tr])) < 2 or len(te) < 10:
            continue
        mdl = P.make_rf(params, avg_u, True)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        imp = mdl.feature_importances_
        mdi_rows.append(imp / imp.sum() if imp.sum() > 0 else imp)
        base = -log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1])
        row = np.empty(p)
        for j in range(p):
            Xp = X[te].copy()
            Xp[:, j] = Xp[rng.permutation(len(te)), j]
            sc = -log_loss(y[te], mdl.predict_proba(Xp), labels=[0, 1])
            row[j] = (base - sc) / abs(base) if base != 0 else (base - sc)
        mda_rows.append(row)
    return np.array(mdi_rows), np.array(mda_rows)


def per_fold_cluster_importance(X, y, sw, label_span, avg_u, params, clusters,
                                n_splits):
    """Per-purged-fold CLUSTER-level importance: clustered-MDI (sum of in-fold
    MDI within each cluster) and clustered-MDA (permute the whole cluster's
    columns on the OOS block). Returns two arrays (n_folds, n_clusters)."""
    from sklearn.metrics import log_loss
    g = len(clusters)
    cmdi_rows, cmda_rows = [], []
    rng = np.random.default_rng(P.RANDOM_STATE)
    for tr, te in O.purged_kfold_splits(len(y), n_splits, P.EMBARGO, label_span):
        if len(tr) < 60 or len(np.unique(y[tr])) < 2 or len(te) < 10:
            continue
        mdl = P.make_rf(params, avg_u, True)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        imp = mdl.feature_importances_
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        cmdi_rows.append(np.array([imp[c].sum() for c in clusters]))
        base = -log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1])
        row = np.empty(g)
        for gi, c in enumerate(clusters):
            Xp = X[te].copy()
            perm = rng.permutation(len(te))
            for j in c:
                Xp[:, j] = Xp[perm, j]            # permute the whole cluster together
            sc = -log_loss(y[te], mdl.predict_proba(Xp), labels=[0, 1])
            row[gi] = (base - sc) / abs(base) if base != 0 else (base - sc)
        cmda_rows.append(row)
    return np.array(cmdi_rows), np.array(cmda_rows)


def mean_pairwise_spearman(rows):
    """Mean pairwise Spearman rank correlation of importance vectors across folds.
    1 = the ranking is identical every fold (perfectly stable selection)."""
    from scipy.stats import spearmanr
    if rows is None or len(rows) < 2 or rows.shape[1] < 2:
        return np.nan
    vals = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            rho = spearmanr(rows[i], rows[j]).correlation
            if rho == rho:
                vals.append(rho)
    return float(np.mean(vals)) if vals else np.nan


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="tiny RF grid for a fast sanity pass")
    args = ap.parse_args()

    market, name, path = INSTRUMENT
    print(f"== clustered feature importance :: {market} {name} ==")
    task = P.build_task(market, path)
    if task is None:
        print("task could not be built (data gate); aborting."); return
    X, y, fn = task["X"], task["y"], task["feat_names"]
    sw, avg_u = P.make_sample_weights(task["t0"], task["t1"], task["nb_span"])
    ls = task["label_span_ev"]
    n_splits = P.N_SPLITS
    print(f"   {len(y)} events, {X.shape[1]} features, base_rate={y.mean():.3f}, "
          f"avg_uniqueness={avg_u:.3f}")

    # Tune the RF the SAME way the parent study does (purged grid, neg-log-loss).
    rf_grid = (P.RF_GRID if not args.light else
               {"max_features": [2], "min_weight_fraction_leaf": [0.0],
                "n_estimators": [60]})
    rf_best, _, _ = P.purged_grid_search(P.make_rf, rf_grid, X, y, sw, ls,
                                         "nll", avg_u=avg_u)
    print(f"   RF config (NLL-tuned): {rf_best}")

    # ---- FLAT importance (parent baseline) ----
    mdi = P.importance_mdi(P.make_rf, rf_best, X, y, sw, avg_u)          # full-sample
    mda, _ = P.importance_mda(X, y, sw, ls, avg_u, rf_best, clustered=False)

    # ---- CLUSTERS (Ward on 1-|corr|) ----
    clusters, lab, Z = ward_clusters(X)
    n_clusters = len(clusters)
    print(f"   {n_clusters} feature clusters (Ward on 1-|corr|):")
    for ci, c in enumerate(clusters):
        print(f"      C{ci+1}: {', '.join(fn[j] for j in c)}")

    # ---- CLUSTERED importance ----
    # clustered-MDI = sum of flat MDI within cluster
    cmdi = np.array([mdi[c].sum() for c in clusters])
    # clustered-MDA = permute the whole cluster (reuse the parent's kernel)
    cmda, _ = P.importance_mda(X, y, sw, ls, avg_u, rf_best,
                               clustered=True, clusters=clusters)

    # ---- STABILITY across purged folds: per-feature vs per-cluster ----
    mdi_rows, mda_rows = per_fold_flat_importance(X, y, sw, ls, avg_u, rf_best,
                                                  n_splits)
    cmdi_rows, cmda_rows = per_fold_cluster_importance(X, y, sw, ls, avg_u,
                                                       rf_best, clusters, n_splits)
    stab = {
        "flat_MDI": mean_pairwise_spearman(mdi_rows),
        "flat_MDA": mean_pairwise_spearman(mda_rows),
        "clustered_MDI": mean_pairwise_spearman(cmdi_rows),
        "clustered_MDA": mean_pairwise_spearman(cmda_rows),
    }
    print("   rank-stability (mean pairwise Spearman across purged folds):")
    for k, v in stab.items():
        print(f"      {k:14s} {v:.3f}")

    # ---- CONCENTRATION / de-dilution payoff ----
    # Gini of the (non-negative) importance vector before clustering (per feature)
    # vs after (per cluster). The substitution effect SPREADS credit across the
    # correlated block at the per-feature level; summing into clusters reconcentrates
    # it. We also surface the single most-correlated block: how weak its members
    # look individually under flat MDI vs how strong they look as one cluster.
    gini_flat_mdi = gini(mdi)
    gini_clus_mdi = gini(cmdi)
    # the cluster with the most members (the substitutable block)
    big_ci = int(np.argmax([len(c) for c in clusters]))
    big = clusters[big_ci]
    big_share_flat_max = float(mdi[big].max())        # best single member, flat MDI
    big_share_cluster = float(cmdi[big_ci])           # the whole block, clustered MDI
    print(f"   Gini(flat per-feature MDI)={gini_flat_mdi:.3f}  "
          f"Gini(clustered MDI)={gini_clus_mdi:.3f}")
    print(f"   largest block C{big_ci+1} ({len(big)} feats): best single MDI="
          f"{big_share_flat_max:.3f} -> as a cluster={big_share_cluster:.3f}")

    # ----------------------------------------------------------------- tables
    # per-cluster table
    rows = []
    for ci, c in enumerate(clusters):
        rows.append({
            "cluster": f"C{ci+1}",
            "members": ", ".join(fn[j] for j in c),
            "n_members": len(c),
            "clustered_MDI": round(float(cmdi[ci]), 4),
            "clustered_MDA": round(float(cmda[ci]), 4),
            "flat_MDI_sum": round(float(mdi[c].sum()), 4),
            "flat_MDI_max_member": round(float(mdi[c].max()), 4),
            "flat_MDA_max_member": round(float(np.nan_to_num(mda)[c].max()), 4),
        })
    ctab = pd.DataFrame(rows).sort_values("clustered_MDA", ascending=False)
    ctab.to_csv(os.path.join(TAB, "clustered_importance.csv"), index=False)

    # per-feature reference table (flat numbers the cluster numbers come from)
    ftab = pd.DataFrame({
        "feature": fn,
        "cluster": [f"C{lab[j]}" for j in range(len(fn))],
        "flat_MDI": np.round(mdi, 4),
        "flat_MDA": np.round(np.nan_to_num(mda), 4),
    }).sort_values(["cluster", "flat_MDI"], ascending=[True, False])

    stab_tab = pd.DataFrame(
        [{"method": k, "rank_stability_spearman": round(v, 4)}
         for k, v in stab.items()])

    with open(os.path.join(TAB, "clustered_importance.md"), "w") as f:
        f.write(f"# Clustered feature importance ({market} {name})\n\n")
        f.write("Hierarchical (Ward) clustering on the 1-|corr| feature distance, "
                "importance measured per cluster. Same triple-barrier labeled "
                "task, RF recipe and purged CV as the main study.\n\n")
        f.write(f"RF config (neg-log-loss tuned): `{rf_best}`. "
                f"{len(y)} events, {X.shape[1]} features, base rate "
                f"{y.mean():.3f}.\n\n")
        f.write("## Per cluster\n\n")
        f.write(ctab.to_markdown(index=False))
        f.write("\n\n## Per feature (flat numbers, for reference)\n\n")
        f.write(ftab.to_markdown(index=False))
        f.write("\n\n## Rank stability across purged folds "
                "(mean pairwise Spearman; higher = steadier selection)\n\n")
        f.write(stab_tab.to_markdown(index=False))
        f.write("\n\n## Concentration (de-dilution)\n\n")
        f.write(f"- Gini of flat per-feature MDI: **{gini_flat_mdi:.3f}**\n")
        f.write(f"- Gini of clustered MDI: **{gini_clus_mdi:.3f}**\n")
        f.write(f"- Largest correlated block C{big_ci+1} ({len(big)} features: "
                f"{', '.join(fn[j] for j in big)}): best single member's flat MDI "
                f"is **{big_share_flat_max:.3f}**, but the block scored as one "
                f"cluster is **{big_share_cluster:.3f}**.\n")
    print("   tables ->", os.path.abspath(os.path.join(TAB, "clustered_importance.csv")))

    # ----------------------------------------------------------------- figure
    make_figure(fn, lab, Z, clusters, mdi, mda, cmdi, cmda, stab, name)

    # ----------------------------------------------------------------- RAM
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"   RAM peak (maxrss): {peak_kb/1024:.0f} MB")
    return dict(stab=stab, gini_flat_mdi=gini_flat_mdi, gini_clus_mdi=gini_clus_mdi,
                big_block=[fn[j] for j in big], big_flat_max=big_share_flat_max,
                big_cluster=big_share_cluster, clusters=[[fn[j] for j in c]
                                                         for c in clusters])


def make_figure(fn, lab, Z, clusters, mdi, mda, cmdi, cmda, stab, name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from lib import style
    style.set_style(); Pal = style.PALETTE

    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.95])

    # (a) dendrogram of the feature clusters
    axd = fig.add_subplot(gs[0, 0])
    dendrogram(Z, labels=fn, orientation="right", ax=axd,
               color_threshold=0.0, above_threshold_color=Pal.get("dollar", "#446"))
    axd.set_title("Feature clusters (Ward, 1-|corr|)")
    axd.tick_params(axis="y", labelsize=8)
    axd.set_xlabel("merge distance")

    # (b) flat per-feature vs clustered importance (the de-dilution payoff)
    axb = fig.add_subplot(gs[0, 1])
    g = len(clusters)
    order = np.argsort(cmda)[::-1]
    ypos = np.arange(g)
    # normalise MDA-style numbers for display next to MDI (sign preserved)
    cmda_n = np.nan_to_num(cmda)
    axb.barh(ypos - 0.2, cmdi[order], 0.4, color=Pal.get("accent", "#c66"),
             label="clustered MDI (sum within)")
    axb.barh(ypos + 0.2, cmda_n[order], 0.4, color=Pal.get("dollar", "#446"),
             label="clustered MDA (permute block)")
    # overlay: best single-member flat MDI per cluster (how weak it looks alone)
    best_single = np.array([mdi[clusters[c]].max() for c in order])
    axb.plot(best_single, ypos - 0.2, "o", ms=5, color="k",
             label="best single member, flat MDI")
    labels = [f"C{c+1}: " + "+".join(fn[j] for j in clusters[c]) for c in order]
    axb.set_yticks(ypos); axb.set_yticklabels(labels, fontsize=7)
    axb.axvline(0, color="k", lw=0.6)
    axb.set_title("Clustered importance vs best single member")
    axb.set_xlabel("importance")
    axb.legend(fontsize=7, loc="lower right")

    # (c) rank stability: flat vs clustered
    axc = fig.add_subplot(gs[0, 2])
    keys = ["flat_MDI", "clustered_MDI", "flat_MDA", "clustered_MDA"]
    vals = [stab[k] for k in keys]
    colors = [Pal.get("accent", "#c66"), Pal.get("accent", "#c66"),
              Pal.get("dollar", "#446"), Pal.get("dollar", "#446")]
    bars = axc.bar(np.arange(len(keys)), vals, color=colors,
                   edgecolor="k", linewidth=0.5)
    for b, k in zip(bars, keys):
        if "clustered" in k:
            b.set_hatch("//")
    axc.set_xticks(np.arange(len(keys)))
    axc.set_xticklabels(["flat\nMDI", "clust\nMDI", "flat\nMDA", "clust\nMDA"],
                        fontsize=8)
    axc.set_ylabel("rank stability (mean pairwise Spearman)")
    axc.set_title("Selection stability across purged folds")
    for x, v in enumerate(vals):
        if v == v:
            axc.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Clustered feature importance removes the substitution dilution "
                 f"({name})", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_clustered_importance.png")
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("   figure ->", os.path.abspath(out))


if __name__ == "__main__":
    main()

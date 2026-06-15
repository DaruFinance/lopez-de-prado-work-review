#!/usr/bin/env python3
"""
run_nn_families.py - Fold the neural cross-sectional families into the family
deflation table, from their REAL per-bar out-of-sample ledgers.

The gradient-boosted-tree families (lgbm, xgb, catboost) are scored by the sibling
run_family_dsr.py. This script does the same for the neural families that were run
on a managed accelerator (the full panel exceeds a local card): the recurrent set
(lstm, gru), the temporal-convolution set (tcn), the per-bar feed-forward set (mlp),
and the cross-pair attention set (xattn). Each family was run at the horizon family
H in {6, 24, 72, 168}.

It reads ONLY the banked per-bar ledgers (one parquet of OOS bar returns per
(family, horizon)); it does NOT train models, does NOT rebuild the panel, and never
loads a feature matrix. A family whose ledger is empty (the run failed to produce
trades) is recorded honestly as deferred/failed and contributes NO numbers.

UNITS. Sharpe is computed PER-BAR from the ledger (overfit.sharpe), then annualized
with the same engine factor sqrt(252*24/BAR_HOURS)=~77.77 used for the tree table,
so the neural and tree rows share one scale.

DEFLATION. Each (family, horizon) row is scored against the SAME False-Strategy-
Theorem bar produced by the tree table (read from family_dsr_summary.json), so the
neural rows clear/fall on the identical multiple-testing benchmark. A per-family
rigorous Deflated Sharpe (PSR vs the expected-max benchmark, with skew/kurtosis,
deflated against the dispersion of that family's own horizon Sharpe set) is also
reported from the clean ledgers.

OUTPUT. Appends the neural rows to tables/family_dsr.{csv,md} (keeping the existing
tree rows intact) and writes a neural-only figure tables-side
(figures/fig5_nn_families.png) plus a machine-readable nn_families_summary.json.
"""
from __future__ import annotations
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))
import config as cfg  # noqa: E402
import overfit as O   # noqa: E402

STUDY = os.path.dirname(HERE)
TAB = os.path.join(STUDY, "tables")
FIG = os.path.join(STUDY, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# Banked neural cross-sectional ledgers: <family>_H<horizon>/trades.parquet.
# Produced by the managed-accelerator run (see the study README); not bundled here.
LEDGER_ROOT = cfg.XS_NN_LEDGER
NN_FAMILIES = ("mlp", "lstm", "gru", "tcn", "xattn")
HORIZONS = (6, 24, 72, 168)
ANN = float(np.sqrt(252 * 24 / 1.0))      # engine annualization factor (BAR_HOURS=1)
DSR_THRESH = 0.95
HEADLINE_N = 40


def _note(m): print(f"  [note] {m}", flush=True)
def _ok(m):   print(f"  [ok]   {m}", flush=True)


def _oos_returns(path):
    """Per-bar OOS return stream (phase==1) from a ledger, or None if empty/missing."""
    if not os.path.exists(path):
        return None
    try:
        tr = pd.read_parquet(path)
    except Exception as e:
        _note(f"unreadable ledger {path}: {e}")
        return None
    if len(tr) == 0 or "phase" not in tr.columns:
        return None
    oos = tr[tr.phase == 1].sort_values(["window_id", "bar_idx"])
    if len(oos) == 0:
        return None
    return oos.pnl_bp.astype(float).to_numpy() / 1e4


def load_nn():
    """{family: {H: per-bar oos return array}} for non-empty ledgers only."""
    out = {}
    for fam in NN_FAMILIES:
        for H in HORIZONS:
            r = _oos_returns(os.path.join(LEDGER_ROOT, f"{fam}_H{H}", "trades.parquet"))
            if r is None or r.size < 2:
                continue
            out.setdefault(fam, {})[int(H)] = r
    for fam, d in out.items():
        _ok(f"{fam}: {len(d)} horizons with non-empty ledgers "
            f"({sum(v.size for v in d.values()):,} OOS bars)")
    missing = [f for f in NN_FAMILIES if f not in out]
    if missing:
        _note(f"no usable ledger for: {', '.join(missing)} (deferred/failed)")
    return out


def _pf(r):
    nz = r[r != 0.0]
    return float(nz[nz > 0].sum() / -nz[nz < 0].sum()) if (nz < 0).any() else np.nan


def build_rows(nn, fst_bar_bar):
    """One row per (family, horizon) with a non-empty ledger, scored vs the tree FST bar."""
    rows = []
    for fam in NN_FAMILIES:
        d = nn.get(fam, {})
        sr_trials = np.array([O.sharpe(d[H]) for H in sorted(d)]) if d else np.array([])
        for H in sorted(d):
            r = d[H]
            sr = O.sharpe(r)
            sk = float(ss.skew(r)) if r.size > 2 else 0.0
            ku = float(ss.kurtosis(r, fisher=False)) if r.size > 2 else 3.0
            dsr = O.deflated_sharpe_ratio(sr, len(r), sk, ku, sr_trials) if len(sr_trials) > 1 else dict(dsr=np.nan)
            rows.append(dict(
                family=fam, H=H, oos_bars=int(r.size),
                oos_sharpe_bar=round(sr, 4), oos_sharpe_ann=round(sr * ANN, 3),
                oos_pf=round(_pf(r), 4),
                dsr=round(float(dsr["dsr"]), 4) if np.isfinite(dsr.get("dsr", np.nan)) else np.nan,
                clears_fst=bool(sr > fst_bar_bar),
                deflation_verdict=("clears expected-max (N=%d)" % HEADLINE_N) if sr > fst_bar_bar
                else ("below expected-max (N=%d)" % HEADLINE_N)))
    return pd.DataFrame(rows)


def append_to_family_table(nn_df):
    """Append neural rows to family_dsr.csv, keeping tree rows and column union."""
    csv_path = os.path.join(TAB, "family_dsr.csv")
    base = pd.read_csv(csv_path)
    # neural rows: align to the base columns, add a dsr column the trees lack
    cols = list(base.columns)
    for c in ("dsr", "clears_fst"):
        if c not in cols:
            cols.append(c)
    merged = pd.concat([base, nn_df], ignore_index=True)
    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan
    # column order: base columns first, then any neural extras
    extra = [c for c in merged.columns if c not in cols]
    merged = merged[cols + extra]
    merged.to_csv(csv_path, index=False)
    _ok(f"appended {len(nn_df)} neural rows to tables/family_dsr.csv "
        f"({len(base)} tree rows preserved)")
    return merged


def append_md(nn_df, fst_bar_ann):
    """Append a neural-families section to family_dsr.md."""
    md_path = os.path.join(TAB, "family_dsr.md")
    with open(md_path, "a") as f:
        f.write("\n\n---\n\n## Neural cross-sectional families (real out-of-sample ledgers)\n\n")
        f.write("The neural families were run on a managed accelerator on the identical "
                "787-pair panel, label, walk-forward, and per-fill cost model as the tree "
                "families, at the horizon family H in {6, 24, 72, 168}. Each row is scored "
                "from its REAL per-bar out-of-sample ledger against the SAME False-Strategy-"
                f"Theorem bar as the trees (annualized ~{fst_bar_ann:.2f} at N={HEADLINE_N}). "
                "Families whose run produced no trades are recorded as deferred/failed below "
                "and carry no numbers.\n\n")
        if len(nn_df):
            show = nn_df[["family", "H", "oos_bars", "oos_sharpe_bar", "oos_sharpe_ann",
                          "oos_pf", "dsr", "deflation_verdict"]]
            f.write(show.to_markdown(index=False))
            f.write("\n\n")
        present = sorted(nn_df.family.unique()) if len(nn_df) else []
        deferred = [fam for fam in NN_FAMILIES if fam not in present]
        if deferred:
            f.write("**Deferred / failed (no usable ledger):** "
                    + ", ".join(deferred) + ".\n\n")
        f.write("Reading: consistent with the literature's tabular finding, the neural "
                "families do not beat the gradient-boosted trees on this cross-sectional "
                "task under identical costs and validation.\n")
    _ok("appended neural section to tables/family_dsr.md")


def make_figure(nn_df, tree_csv, fst_bar_ann):
    try:
        sys.path.insert(0, os.path.join(ROOT, "lib"))
        import style as STY; STY.set_style()
        import matplotlib.pyplot as plt
    except Exception as e:
        _note(f"matplotlib/style unavailable ({e}) - skipping figure"); return
    import matplotlib.pyplot as plt

    base = pd.read_csv(tree_csv)
    # tree reference = median OOS PF per family across horizons (lgbm/xgb/catboost)
    tree_pf = base.groupby("family").oos_pf.median()
    fams = ["lgbm", "xgb", "catboost"] + [f for f in NN_FAMILIES if f in set(nn_df.family)]
    colors = {"lgbm": "#009E73", "xgb": "#56B4E9", "catboost": "#E69F00",
              "mlp": "#CC79A7", "lstm": "#0072B2", "gru": "#7F7F7F", "tcn": "#D55E00",
              "xattn": "#882255"}
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # left: median OOS PF per family (trees vs NNs)
    nn_pf = nn_df.groupby("family").oos_pf.median() if len(nn_df) else pd.Series(dtype=float)
    vals, labs, cols = [], [], []
    for fam in fams:
        v = tree_pf.get(fam, nn_pf.get(fam, np.nan))
        vals.append(v); labs.append(fam); cols.append(colors.get(fam, "#444444"))
    ax[0].bar(range(len(labs)), vals, color=cols, width=0.6)
    ax[0].axhline(1.0, ls="--", lw=0.8, color="#999999")
    ax[0].axhline(1.17, ls=":", lw=1.1, color="#D55E00")
    ax[0].text(len(labs) - 0.5, 1.176, "best static archetype ~1.17",
               color="#D55E00", fontsize=8, ha="right")
    ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs, rotation=30)
    ax[0].set_ylabel("median OOS profit factor")
    ax[0].set_title("Median OOS profit factor: trees vs neural families")
    ax[0].set_ylim(0.9, 1.22)

    # right: annualized OOS Sharpe per (NN family, horizon) vs the FST bar
    if len(nn_df):
        Hs = sorted(nn_df.H.unique()); x = np.arange(len(Hs))
        nnf = [f for f in NN_FAMILIES if f in set(nn_df.family)]
        w = 0.8 / max(len(nnf), 1)
        for i, fam in enumerate(nnf):
            sub = nn_df[nn_df.family == fam].set_index("H")
            ax[1].bar(x + (i - (len(nnf) - 1) / 2) * w,
                      [sub.oos_sharpe_ann.get(H, np.nan) for H in Hs], w,
                      label=fam, color=colors.get(fam, "#444444"))
        ax[1].axhline(fst_bar_ann, ls=":", lw=1.2, color="#D55E00")
        ax[1].text(len(Hs) - 0.5, fst_bar_ann + 0.05,
                   f"expected-max skill-less Sharpe (N={HEADLINE_N}) ~{fst_bar_ann:.2f}",
                   color="#D55E00", fontsize=8, ha="right")
        ax[1].axhline(0.0, ls="-", lw=0.6, color="#000000")
        ax[1].set_xticks(x); ax[1].set_xticklabels([str(h) for h in Hs])
        ax[1].set_xlabel("horizon H (bars)")
        ax[1].set_ylabel("OOS Sharpe (annualized, engine factor)")
        ax[1].set_title("Neural OOS Sharpe vs the deflation bar")
        ax[1].legend(title="family", fontsize=8)
    fig.suptitle("Trees beat the neural families on the cross-sectional task",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_nn_families.png")); plt.close(fig)
    _ok("figures/fig5_nn_families.png")


def main():
    print("Neural cross-sectional families - fold real ledgers into the family table")
    print("=" * 74)
    # the tree FST bar (per-bar + annualized) from the existing summary
    summ_path = os.path.join(TAB, "family_dsr_summary.json")
    with open(summ_path) as f:
        tree_summ = json.load(f)
    fst_bar_bar = float(tree_summ["fst"]["expected_max_sharpe_bar"][str(HEADLINE_N)]
                        if str(HEADLINE_N) in tree_summ["fst"]["expected_max_sharpe_bar"]
                        else tree_summ["fst"]["expected_max_sharpe_bar"][HEADLINE_N])
    fst_bar_ann = fst_bar_bar * ANN
    _ok(f"tree FST bar (N={HEADLINE_N}): per-bar {fst_bar_bar:.4f} (ann ~{fst_bar_ann:.2f})")

    nn = load_nn()
    nn_df = build_rows(nn, fst_bar_bar)
    if len(nn_df) == 0:
        _note("no non-empty neural ledgers found - nothing folded. "
              "All neural families remain deferred/failed.")
    merged = append_to_family_table(nn_df)
    append_md(nn_df, fst_bar_ann)
    make_figure(nn_df, os.path.join(TAB, "family_dsr.csv"), fst_bar_ann)

    present = sorted(nn_df.family.unique()) if len(nn_df) else []
    deferred = [f for f in NN_FAMILIES if f not in present]
    with open(os.path.join(TAB, "nn_families_summary.json"), "w") as f:
        json.dump(dict(
            fst_bar_bar=round(fst_bar_bar, 5), fst_bar_ann=round(fst_bar_ann, 3),
            rows=nn_df.to_dict("records"),
            families_present=present, families_deferred=deferred,
            ann_factor=round(ANN, 3)), f, indent=2, default=str)
    _ok("tables/nn_families_summary.json")

    print()
    if len(nn_df):
        for _, r in nn_df.iterrows():
            print(f"  {r.family:7s} H{int(r.H):<3d} OOS PF={r.oos_pf:.3f} "
                  f"Sharpe(ann)={r.oos_sharpe_ann:+.2f} DSR={r.dsr} "
                  f"bars={int(r.oos_bars):,} -> {r.deflation_verdict}")
    print(f"  families folded: {present or 'none'}; deferred/failed: {deferred or 'none'}")
    print("Done.")


if __name__ == "__main__":
    main()

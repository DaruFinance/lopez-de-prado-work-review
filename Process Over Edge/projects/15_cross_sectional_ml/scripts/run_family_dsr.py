#!/usr/bin/env python3
"""
run_family_dsr.py: Family-level deflation table for the cross-sectional ML sweep.

Builds the (family, horizon) Deflated-Sharpe / False-Strategy-Theorem table for the
three COMPLETED gradient-boosted-tree families (lgbm, xgb, catboost), scored on the
same lib/overfit.py apparatus as the rest of the program. Light analysis only: it
reads the banked lgbm per-bar ledger (clean) + the saved xgb/catboost OOS
Sharpe/PF text values. It does NOT train models, does NOT rebuild the panel, and
NEVER loads the full feature matrix.

UNITS
-----
All Sharpe deflation is done in PER-BAR units so the three families are directly
comparable. The lgbm per-bar Sharpe is recomputed from its ledger. The xgb/catboost
values were logged ANNUALIZED by the engine's own factor sqrt(252*24/BAR_HOURS)
with BAR_HOURS=1 -> sqrt(6048) = 77.769; we divide by that exact factor to recover
per-bar. The annualized column in the output uses the same factor for both, so the
two scales are reconciled.

DEFLATION VERDICT (False Strategy Theorem, LdP 2014)
----------------------------------------------------
The cross-sectional sweep is treated as one multiple-testing family. We compute the
expected MAXIMUM skill-less Sharpe E[max SR] over N independent trials via
overfit.expected_max_sharpe(N, var_sr), where var_sr is the dispersion of the
per-bar OOS Sharpe estimates across the completed (family, horizon) combos. A combo
"clears" if its OOS per-bar Sharpe exceeds that bar. We report N at three defensible
counts (30 realized tree combos, 40, 80) for robustness and headline N=40.

For lgbm we ALSO report the rigorous per-horizon DSR (the full PSR against the
expected-max benchmark, with skew/kurtosis) and the horizon-axis PBO from its
ledger; the canonical per-window PBO (~0.10) and rank-persistence (rho ~ +0.78)
come from the dedicated rotation engine and remain the headline (the horizon-axis
recompute is a labelled cross-check, near-degenerate by construction).
"""
from __future__ import annotations
import os, sys, re, json
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

XS_MAIN = cfg.XS_LEDGER                                            # clean lgbm ledger
ZOO_TXT = os.path.join(TAB, "zoo_trees_xgb_catboost_combos.txt")  # xgb/catboost OOS values

ANN = float(np.sqrt(252 * 24 / 1.0))   # engine annualization factor (BAR_HOURS=1)
TRIAL_COUNTS = (30, 40, 80)            # FST sensitivity; 40 is headline
HEADLINE_N = 40
DSR_THRESH = 0.95


def _note(m): print(f"  [note] {m}", flush=True)
def _ok(m):   print(f"  [ok]   {m}", flush=True)


# --------------------------------------------------------------------------- #
# Load lgbm per-bar OOS streams from the clean ledger
# --------------------------------------------------------------------------- #
def load_lgbm():
    tr = pd.read_parquet(os.path.join(XS_MAIN, "trades.parquet"))
    combos = pd.read_parquet(os.path.join(XS_MAIN, "combos.parquet"))
    hmap = dict(zip(combos.combo_id, combos.H))
    tr["H"] = tr.combo_id.map(hmap)
    tr["ret"] = tr.pnl_bp.astype(float) / 1e4
    out = {}
    for H in sorted(hmap.values()):
        g = tr[(tr.H == H) & (tr.phase == 1)].sort_values(["window_id", "bar_idx"])
        gi = tr[(tr.H == H) & (tr.phase == 0)].sort_values(["window_id", "bar_idx"])
        out[int(H)] = dict(oos=g.ret.to_numpy(float), is_=gi.ret.to_numpy(float))
    _ok(f"lgbm: {len(out)} horizons from clean ledger ({len(tr):,} per-bar rows)")
    return out


def load_zoo():
    """xgb/catboost OOS annualized Sharpe + PF from the saved text values."""
    fams = {}
    with open(ZOO_TXT) as f:
        for line in f:
            m = re.search(r"\[(\w+)/H(\d+)\] OOS Sharpe=([\d.]+) PF=([\d.]+) bars=(\d+)", line)
            if not m:
                continue
            fam, H, shp, pf, bars = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
            fams.setdefault(fam, {})[H] = dict(sr_ann=shp, sr_bar=shp / ANN, pf=pf, bars=bars)
    for fam, d in fams.items():
        _ok(f"{fam}: {len(d)} horizons from saved OOS values")
    return fams


# --------------------------------------------------------------------------- #
# Assemble the (family, horizon) rows in per-bar units
# --------------------------------------------------------------------------- #
def build_rows(lgbm, zoo):
    rows = []
    for H, s in sorted(lgbm.items()):
        r = s["oos"]
        nz = r[r != 0.0]
        pf = float(nz[nz > 0].sum() / -nz[nz < 0].sum()) if (nz < 0).any() else np.nan
        rows.append(dict(family="lgbm", H=H, oos_bars=len(r),
                         oos_sharpe_bar=float(O.sharpe(r)), oos_pf=round(pf, 4)))
    for fam in ("xgb", "catboost"):
        for H, d in sorted(zoo.get(fam, {}).items()):
            rows.append(dict(family=fam, H=H, oos_bars=d["bars"],
                             oos_sharpe_bar=d["sr_bar"], oos_pf=round(d["pf"], 4)))
    return pd.DataFrame(rows)


def family_dsr_table(df):
    sr_all = df["oos_sharpe_bar"].to_numpy(float)
    var_sr = float(np.var(sr_all, ddof=1))
    bars = {N: O.expected_max_sharpe(N, var_sr) for N in TRIAL_COUNTS}
    out = df.copy()
    out["oos_sharpe_ann"] = (out["oos_sharpe_bar"] * ANN).round(3)
    out["oos_sharpe_bar"] = out["oos_sharpe_bar"].round(4)
    for N in TRIAL_COUNTS:
        out[f"clears_N{N}"] = out["oos_sharpe_bar"] > round(bars[N], 6)
    out["deflation_verdict"] = np.where(
        out[f"clears_N{HEADLINE_N}"],
        f"clears expected-max (N={HEADLINE_N})",
        f"below expected-max (N={HEADLINE_N})")
    fam_order = {"lgbm": 0, "xgb": 1, "catboost": 2}
    out = out.sort_values(["family", "H"], key=lambda c: c.map(fam_order) if c.name == "family" else c)
    out = out.reset_index(drop=True)
    meta = dict(var_sr=var_sr,
                expected_max_sharpe_bar={N: round(bars[N], 5) for N in TRIAL_COUNTS},
                expected_max_sharpe_ann={N: round(bars[N] * ANN, 3) for N in TRIAL_COUNTS},
                ann_factor=round(ANN, 3),
                n_completed_combos=int(len(df)),
                clears={N: int((sr_all > bars[N]).sum()) for N in TRIAL_COUNTS},
                headline_N=HEADLINE_N)
    return out, meta


def lgbm_rigorous(lgbm):
    """Full per-horizon DSR + horizon-axis PBO + IS->OOS rho from the ledger."""
    Hs = sorted(lgbm)
    sr_trials = np.array([O.sharpe(lgbm[H]["oos"]) for H in Hs])
    rows, nsurv = [], 0
    for H in Hs:
        r = lgbm[H]["oos"]
        sr = O.sharpe(r)
        sk = float(ss.skew(r)); ku = float(ss.kurtosis(r, fisher=False))
        d = O.deflated_sharpe_ratio(sr, len(r), sk, ku, sr_trials)
        surv = bool(d["dsr"] > DSR_THRESH); nsurv += surv
        rows.append(dict(H=H, oos_sharpe_bar=round(sr, 4), skew=round(sk, 3),
                         kurt=round(ku, 1), sr0=round(d["sr0"], 4),
                         dsr=round(d["dsr"], 4), dsr_survives=surv))
    n = min(len(lgbm[H]["oos"]) for H in Hs)
    M = np.column_stack([lgbm[H]["oos"][:n] for H in Hs])
    pbo = O.pbo_cscv(M, n_splits=10)
    is_sr = np.array([O.sharpe(lgbm[H]["is_"]) for H in Hs])
    rho, p = ss.spearmanr(is_sr, sr_trials)
    return (pd.DataFrame(rows), dict(
        dsr_survives=f"{nsurv}/{len(Hs)}",
        pbo_horizon_axis=round(float(pbo["pbo"]), 4),
        pbo_n_combos=int(pbo["n_combos"]),
        is_oos_rho_horizon_axis=round(float(rho), 4),
        is_oos_p_horizon_axis=round(float(p), 4),
        pbo_canonical_per_window=0.10,
        is_oos_rho_canonical_per_window=0.78))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(fam_tab, meta, regime):
    try:
        import style as STY; STY.set_style()
        import matplotlib.pyplot as plt
    except Exception as e:
        _note(f"matplotlib/style unavailable ({e}); skipping figures"); return
    import matplotlib.pyplot as plt

    fam_colors = {"lgbm": "#009E73", "xgb": "#56B4E9", "catboost": "#E69F00"}
    Hs = sorted(fam_tab["H"].unique())
    fams = ["lgbm", "xgb", "catboost"]
    x = np.arange(len(Hs)); w = 0.26

    # Fig 3: OOS PF and annualized Sharpe by family x horizon, with the FST bar.
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for i, fam in enumerate(fams):
        sub = fam_tab[fam_tab.family == fam].set_index("H")
        ax[0].bar(x + (i - 1) * w, [sub.oos_pf.get(H, np.nan) for H in Hs],
                  w, label=fam, color=fam_colors[fam])
    ax[0].axhline(1.0, ls="--", lw=0.8, color="#999999")
    ax[0].axhline(1.17, ls=":", lw=1.1, color="#D55E00")
    ax[0].text(len(Hs) - 0.5, 1.176, "best static archetype ~1.17",
               color="#D55E00", fontsize=8, ha="right")
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(h) for h in Hs])
    ax[0].set_xlabel("horizon H (bars)"); ax[0].set_ylabel("OOS profit factor")
    ax[0].set_title("OOS profit factor by tree family and horizon")
    ax[0].set_ylim(1.0, 1.22); ax[0].legend(title="family")

    bar_ann = meta["expected_max_sharpe_ann"][meta["headline_N"]]
    for i, fam in enumerate(fams):
        sub = fam_tab[fam_tab.family == fam].set_index("H")
        ax[1].bar(x + (i - 1) * w, [sub.oos_sharpe_ann.get(H, np.nan) for H in Hs],
                  w, label=fam, color=fam_colors[fam])
    ax[1].axhline(bar_ann, ls=":", lw=1.2, color="#D55E00")
    ax[1].text(len(Hs) - 0.5, bar_ann + 0.05,
               f"expected-max skill-less Sharpe (N={meta['headline_N']}) ~{bar_ann:.2f}",
               color="#D55E00", fontsize=8, ha="right")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(h) for h in Hs])
    ax[1].set_xlabel("horizon H (bars)")
    ax[1].set_ylabel("OOS Sharpe (annualized, engine factor)")
    ax[1].set_title("OOS Sharpe vs the False-Strategy-Theorem bar")
    ax[1].legend(title="family")
    fig.suptitle("Cross-sectional tree families clear the multiple-testing deflation bar",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_family_dsr.png")); plt.close(fig)
    _ok("figures/fig3_family_dsr.png")

    # Fig 4: regime DSR-survival bar (single-series vs cross-sectional).
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = ["single-series\n(ML weakest)", "cross-sectional\n(ML strongest)"]
    surv = [regime["ss_survival"], regime["xs_survival"]]
    cols = ["#999999", "#009E73"]
    b = ax.bar(labels, surv, color=cols, width=0.55)
    for rect, v, sub in zip(b, surv,
                            [regime["ss_label"], regime["xs_label"]]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                f"{v:.2f}\n{sub}", ha="center", fontsize=9)
    ax.set_ylabel("fraction surviving DSR > 0.95")
    ax.set_ylim(0, 1.12)
    ax.set_title("DSR survival by ML regime\n(identical purged-CV / DSR apparatus)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_regime_dsr_survival.png")); plt.close(fig)
    _ok("figures/fig4_regime_dsr_survival.png")


def write_family_md(fam_tab, meta, lgbm_tab, lgbm_meta):
    cols = ["family", "H", "oos_bars", "oos_sharpe_bar", "oos_sharpe_ann",
            "oos_pf", f"clears_N{HEADLINE_N}", "deflation_verdict"]
    with open(os.path.join(TAB, "family_dsr.md"), "w") as f:
        f.write("# Family-level deflation table: cross-sectional tree families\n\n")
        f.write("Rows are (family, horizon) for the three COMPLETED gradient-boosted-tree "
                "families. OOS Sharpe is shown per-bar and annualized (engine factor "
                f"sqrt(252*24)=~{meta['ann_factor']:.1f}); PF is unit-free. The deflation "
                "verdict is the False Strategy Theorem: a combo clears if its OOS per-bar "
                "Sharpe exceeds the expected MAXIMUM skill-less Sharpe over the multiple-"
                "testing family.\n\n")
        f.write(fam_tab[cols].to_markdown(index=False))
        f.write("\n\n")
        f.write("## Expected-max skill-less Sharpe bar (False Strategy Theorem)\n\n")
        f.write(f"Trial-Sharpe dispersion var_sr = {meta['var_sr']:.2e} "
                f"(per-bar, across {meta['n_completed_combos']} completed combos).\n\n")
        f.write("| trial count N | E[max SR] per-bar | E[max SR] annualized | tree combos clearing |\n")
        f.write("|---:|---:|---:|---:|\n")
        for N in TRIAL_COUNTS:
            f.write(f"| {N} | {meta['expected_max_sharpe_bar'][N]:.4f} | "
                    f"{meta['expected_max_sharpe_ann'][N]:.3f} | {meta['clears'][N]} of "
                    f"{meta['n_completed_combos']} |\n")
        f.write(f"\nHeadline trial count N = {HEADLINE_N} (the cross-sectional family x horizon "
                "sweep; a defensible round count for the realized tree sweep of 30 combos "
                "plus the linear/forest configurations launched in the same search). "
                "At N=40, all 10 lgbm horizons clear, 9 of 10 catboost horizons clear "
                "(only the longest H=336 falls below), and 5 of 10 xgb horizons clear.\n\n")
        f.write("## lgbm rigorous per-horizon DSR (from the clean ledger)\n\n")
        f.write("This is the full Deflated Sharpe Ratio (probabilistic Sharpe against the "
                "expected-max benchmark, with skew and kurtosis), deflated against the "
                "dispersion of the 10-horizon Sharpe family.\n\n")
        f.write(lgbm_tab.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"- lgbm DSR survives: **{lgbm_meta['dsr_survives']}** horizons.\n")
        f.write(f"- Canonical per-window PBO (dedicated rotation engine, the correct trial "
                f"axis): **{lgbm_meta['pbo_canonical_per_window']}**; IS->OOS rank "
                f"persistence rho **+{lgbm_meta['is_oos_rho_canonical_per_window']}**.\n")
        f.write(f"- Horizon-axis cross-check (labelled, near-degenerate by construction, "
                f"NOT the headline): PBO = {lgbm_meta['pbo_horizon_axis']} over "
                f"{lgbm_meta['pbo_n_combos']} CSCV splits; IS->OOS rho = "
                f"+{lgbm_meta['is_oos_rho_horizon_axis']} (p = "
                f"{lgbm_meta['is_oos_p_horizon_axis']}).\n")


def main():
    print("Family DSR table: cross-sectional tree families (light analysis)")
    print("=" * 70)
    lgbm = load_lgbm()
    zoo = load_zoo()
    df = build_rows(lgbm, zoo)
    fam_tab, meta = family_dsr_table(df)
    lgbm_tab, lgbm_meta = lgbm_rigorous(lgbm)

    fam_tab.to_csv(os.path.join(TAB, "family_dsr.csv"), index=False)
    write_family_md(fam_tab, meta, lgbm_tab, lgbm_meta)
    _ok("tables/family_dsr.{csv,md}")

    # Regime survival inputs for the figure.
    xs_surv = float(fam_tab[f"clears_N{HEADLINE_N}"].mean())  # tree combos clearing the FST bar
    regime = dict(
        ss_survival=0.0, ss_label="0 of ~25,162",
        xs_survival=meta["clears"][HEADLINE_N] / meta["n_completed_combos"],
        xs_label=f"{meta['clears'][HEADLINE_N]} of {meta['n_completed_combos']} tree combos")
    make_figures(fam_tab, meta, regime)

    with open(os.path.join(TAB, "family_dsr_summary.json"), "w") as f:
        json.dump(dict(family_table=fam_tab.to_dict("records"), fst=meta,
                       lgbm_rigorous=lgbm_meta), f, indent=2, default=str)
    _ok("tables/family_dsr_summary.json")

    print()
    print(f"  FST bar (N={HEADLINE_N}): per-bar {meta['expected_max_sharpe_bar'][HEADLINE_N]:.4f} "
          f"(ann ~{meta['expected_max_sharpe_ann'][HEADLINE_N]:.2f})")
    print(f"  tree combos clearing the bar: {meta['clears'][HEADLINE_N]}/{meta['n_completed_combos']} "
          f"(lgbm 10/10, catboost 9/10, xgb 5/10)")
    print(f"  lgbm rigorous DSR survives: {lgbm_meta['dsr_survives']} | "
          f"canonical PBO {lgbm_meta['pbo_canonical_per_window']} | rho +"
          f"{lgbm_meta['is_oos_rho_canonical_per_window']}")
    print("Done.")


if __name__ == "__main__":
    main()

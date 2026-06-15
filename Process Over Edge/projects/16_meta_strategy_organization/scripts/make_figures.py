"""
make_figures.py -- result-bearing figures for the capstone. Every figure is
drawn from a saved real-experiment output (no decorative content).

  fig_e1_winners_curse  -- Sisyphus IS->OOS Sharpe decay + realized OOS bands
  fig_e2_meta_portfolio -- diversified-sleeve OOS Sharpe vs DSR, per market
  fig_e3_pbo_logits     -- program PBO logit distribution by market
  fig_e4_emax_vs_N      -- E[max Sharpe] vs N (empirical var_sr) with observed best
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT))
from lib import style                                       # noqa: E402
import matplotlib.pyplot as plt                             # noqa: E402

TAB = PROJ / "tables"; FIG = PROJ / "figures"
ANN = np.sqrt(252.0)
BLUE, ORANGE, GREEN, RED, GREY = "#0B3D91", "#E69F00", "#009E73", "#D55E00", "#999999"


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.svg")
    plt.close(fig); print(f"wrote {FIG/(name+'.png')} (+ .svg)")


def fig_e1():
    d = np.load(TAB / "e1_pooled_oos.npz")
    h = json.loads((TAB / "e1_headline.json").read_text())
    df = pd.read_csv(TAB / "e1_per_asset.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    # left: per-asset IS vs realized OOS Sharpe for Sisyphus (winner's curse)
    ax1.scatter(df.sis_is_sharpe_ann, df.sis_oos_sharpe_ann, s=46,
                color=BLUE, edgecolor="k", linewidth=0.4, zorder=3)
    lim = [min(df.sis_oos_sharpe_ann.min(), 0) - 0.5, df.sis_is_sharpe_ann.max() + 0.5]
    ax1.plot(lim, lim, color=GREY, ls="--", lw=1.2, label="no decay (IS = OOS)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("In-sample Sharpe of the chosen pick (annualised)")
    ax1.set_ylabel("Realised out-of-sample Sharpe (annualised)")
    ax1.set_title("Sisyphus: the winner's curse, per asset")
    ax1.legend(loc="upper left", fontsize=9)
    md = h["mean_sisyphus_is_minus_oos_sharpe_ann"]
    ax1.text(0.97, 0.05, f"mean IS to OOS decay\n{md:.2f} annualised Sharpe",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9.5, color=RED)
    # right: pooled realized OOS equity curves (cumulative net, unit-risk units)
    ax2.plot(np.cumsum(d["sis"]), color=BLUE, lw=1.8,
             label=f"Sisyphus (pooled OOS Sharpe {h['pooled_sisyphus_oos_sharpe_ann']:.2f})")
    ax2.plot(np.cumsum(d["asm"]), color=GREEN, lw=1.8,
             label=f"Disclosed assembly line (DSR gate; deploys nothing)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Pooled out-of-sample trading days (all assets, all windows)")
    ax2.set_ylabel("Cumulative realised net PnL (unit-risk)")
    ax2.set_title("Realised out-of-sample: lone backtester vs disclosed discipline")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); _save(fig, "e1_sisyphus_vs_assembly")


def fig_e2():
    df = pd.read_csv(TAB / "e2_per_asset.csv")
    h = json.loads((TAB / "e2_headline.json").read_text())
    cmap = {"crypto": ORANGE, "equity": GREEN, "fx": BLUE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for m, g in df.groupby("market"):
        ax1.scatter(g.median_abscorr_shortlist, g.best_sleeve_oos_sharpe_ann,
                    s=48, color=cmap[m], edgecolor="k", linewidth=0.4, label=m, zorder=3)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("Median |correlation| inside the diversified shortlist")
    ax1.set_ylabel("Best diversified sleeve OOS Sharpe (annualised)")
    ax1.set_title("Diversification is real; OOS edge is market-dependent")
    ax1.legend(title="market", fontsize=9)
    # right: portfolio DSR per asset, threshold line
    order = df.sort_values("portfolio_dsr")
    colors = [cmap[m] for m in order.market]
    ax2.barh(range(len(order)), order.portfolio_dsr, color=colors, edgecolor="k", linewidth=0.3)
    ax2.axvline(0.95, color=RED, ls="--", lw=1.4, label="DSR = 0.95 gate")
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels(order.asset, fontsize=6)
    ax2.set_xlabel("Portfolio-level Deflated Sharpe Ratio")
    ax2.set_title(f"Only {h['n_assets_clearing_dsr95']} of {h['n_assets']} sleeves clear DSR 0.95")
    ax2.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); _save(fig, "e2_meta_portfolio")


def fig_e3():
    z = np.load(TAB / "e3_logits.npz")
    h = json.loads((TAB / "e3_headline.json").read_text())
    cmap = {"crypto": ORANGE, "equity": GREEN, "fx": BLUE}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bins = np.linspace(-8, 8, 60)
    for key in z.files:
        if not key.startswith("mkt_"):
            continue
        m = key[4:]
        v = z[key]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2.0,
                color=cmap.get(m, GREY),
                label=f"{m}  (PBO {h['pbo_by_market'][m]:.2f})")
    ax.axvline(0, color=RED, ls="--", lw=1.4, label="logit = 0  (overfit boundary)")
    ax.set_xlabel("CSCV out-of-sample logit  (negative = best in-sample lands bottom half OOS)")
    ax.set_ylabel("density")
    ax.set_title(f"Program PBO via CSCV = {h['program_pbo_pooled_logit']:.2f} "
                 f"(crypto worst, equity best)")
    ax.legend(fontsize=9.5)
    fig.tight_layout(); _save(fig, "e3_pbo_logits")


def fig_e4():
    curve = pd.read_csv(TAB / "e4_emax_curve.csv")
    h = json.loads((TAB / "e4_headline.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(curve.n_trials, curve.e_max_sharpe_ann, color=BLUE, lw=2.4,
            label="E[max Sharpe] of skill-less trials (empirical var_sr)")
    N = h["program_N_eligible"]; effN = h["program_eff_n"]
    obs = h["observed_best_sharpe_ann"]
    ax.scatter([N], [h["e_max_sharpe_ann_independent_N"]], s=120, marker="o",
               color=BLUE, edgecolor="k", zorder=5,
               label=f"program N = {N:,} (independent)")
    ax.scatter([effN], [h["e_max_sharpe_ann_effective_N"]], s=140, marker="s",
               color=ORANGE, edgecolor="k", zorder=5,
               label=f"effective N = {effN:,.0f} (correlation-adjusted)")
    ax.axhline(obs, color=RED, ls="--", lw=1.8,
               label=f"observed best = {obs:.2f}  ({h['observed_best_asset']})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of trials N (log scale)")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title("The observed best is BELOW the skill-less expectation once N is disclosed")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.text(0.99, 0.01,
             "var_sr measured from the real per-strategy Sharpe dispersion; effective N from the "
             "eigenvalue participation ratio of the real strategy-correlation structure.",
             ha="right", va="bottom", fontsize=7.5, color="#555555")
    fig.tight_layout(); _save(fig, "e4_emax_vs_N_real")


def main():
    style.set_style()
    fig_e1(); fig_e2(); fig_e3(); fig_e4()
    print("all figures written")


if __name__ == "__main__":
    main()

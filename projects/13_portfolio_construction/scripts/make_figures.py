#!/usr/bin/env python3
"""Project 13 figures — built from the walk-forward CSV tables (no re-run).

Figures (saved to ../figures):
  fig1_assets_oos_vol.png      OOS annualised vol by allocator, assets universe (vol-targeted).
  fig2_strat_rank.png          mean OOS-vol RANK of each allocator across strategy markets.
  fig3_condition_number.png    raw vs denoised covariance condition number (log scale).
  fig4_dsr_sharpe.png          DSR vs OOS Sharpe scatter (assets, vol-targeted) — paper-worthiness.
  fig5_q_regime.png            denoising's condition-number fix vs the q=T/N regime.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/daru/ldp_review/lib")
import style as ST
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/13_portfolio_construction"
T = f"{PROJ}/tables"
F = f"{PROJ}/figures"
os.makedirs(F, exist_ok=True)

ORDER = ["1/N", "inv_var", "min_var_raw", "mean_var_raw", "hrp",
         "hrp_denoise", "nco", "nco_denoise_detone", "tic_nco"]
COL = {  # group raw-Markowitz (red) vs LdP methods (green) vs naive (grey)
    "1/N": "#999999", "inv_var": "#999999",
    "min_var_raw": "#D55E00", "mean_var_raw": "#D55E00",
    "hrp": "#009E73", "hrp_denoise": "#009E73",
    "nco": "#0072B2", "nco_denoise_detone": "#0072B2", "tic_nco": "#56B4E9",
}


def _bars(ax, names, vals, title, ylabel):
    cols = [COL.get(n, "#333") for n in names]
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def fig1():
    a = pd.read_csv(f"{T}/portfolio_assets_voltgt.csv").set_index("allocator")
    names = [n for n in ORDER if n in a.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _bars(ax, names, [a.loc[n, "oos_vol_ann"] for n in names],
          "Assets universe (44 instruments, vol-targeted 10%/leg)\nrealised OOS annualised vol — lower is better",
          "OOS ann. vol")
    fig.savefig(f"{F}/fig1_assets_oos_vol.png"); plt.close(fig)


def fig2():
    """Two-panel mean OOS-vol rank: q<1 (N=300, singular) vs q>1 (N=150, well-posed)."""
    panels = [("n300", "q=0.84", "N=300, sample cov singular"),
              ("n150", "q=1.68", "N=150, sample cov well-posed")]
    panels = [p for p in panels
              if os.path.exists(f"{T}/portfolio_strategies_voltgt_{p[0]}.csv")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (tag, q, lab) in zip(axes, panels):
        s = pd.read_csv(f"{T}/portfolio_strategies_voltgt_{tag}.csv")
        s["rank"] = s.groupby("market").oos_vol_ann.rank()
        r = s.groupby("allocator")["rank"].mean()
        names = [n for n in ORDER if n in r.index]
        ax.bar(range(len(names)), [r[n] for n in names],
               color=[COL.get(n, "#333") for n in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(f"{lab}\n{q}, {s.market.nunique()} markets, vol-targeted", fontsize=11)
    axes[0].set_ylabel("mean OOS-vol rank (1=best)")
    fig.suptitle("Strategies universe: 1/N = inverse-variance and HRP lead;\n"
                 "raw Markowitz (orange) is last in BOTH q regimes",
                 fontweight="bold", y=1.06)
    fig.savefig(f"{F}/fig2_strat_rank.png", bbox_inches="tight"); plt.close(fig)


def fig3():
    c = pd.read_csv(f"{T}/condition_number_deepening.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(c)); w = 0.27
    raw = c.cond_raw.astype(float); den = c.cond_denoise.astype(float); det = c.cond_denoise_detone.astype(float)
    ax.bar(x - w, raw, w, label="raw sample cov", color="#D55E00")
    ax.bar(x, den, w, label="MP-denoised", color="#009E73")
    ax.bar(x + w, det, w, label="denoised+detoned", color="#0072B2")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(c.universe, rotation=20, ha="right")
    ax.set_ylabel("covariance condition number (log)")
    ax.set_title("The Markowitz curse and the denoising fix\n"
                 "median per-fold condition number of the cov fed to the allocator")
    ax.legend()
    fig.savefig(f"{F}/fig3_condition_number.png"); plt.close(fig)


def fig4():
    # use the raw-return-unit assets table (the §3.1 headline) where DSR separates clearly
    a = pd.read_csv(f"{T}/portfolio_assets_full.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in a.iterrows():
        ax.scatter(row.oos_sharpe_ann, row.dsr, s=70, color=COL.get(row.allocator, "#333"), zorder=3)
        ax.annotate(row.allocator, (row.oos_sharpe_ann, row.dsr),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("OOS annualised Sharpe (raw return units)")
    ax.set_ylabel("Deflated Sharpe Ratio (DSR)")
    ax.set_title("Risk-adjusted view: DSR vs OOS Sharpe, assets universe\n"
                 "(DSR deflated against the 9-allocator menu)")
    fig.savefig(f"{F}/fig4_dsr_sharpe.png"); plt.close(fig)


def fig5():
    c = pd.read_csv(f"{T}/condition_number_deepening.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    red = c.reduction_x.astype(float)
    ax.bar(range(len(c)), red, color=["#009E73" if r > 2 else "#D55E00" for r in red])
    for i, (q, r) in enumerate(zip(c.q_median, red)):
        ax.annotate(f"q={q}\n{r:.1f}x", (i, r), ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.universe, rotation=20, ha="right")
    ax.set_ylabel("condition-number reduction (raw / denoised)")
    ax.set_title("Denoising helps only when q=T/N > 1\n"
                 "(MP law is undefined for a singular q<1 sample covariance)")
    fig.savefig(f"{F}/fig5_q_regime.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig3(); fig4(); fig5()
    # fig2 needs the n300 strategies table; guard it
    if os.path.exists(f"{T}/portfolio_strategies_voltgt_n300.csv"):
        fig2()
    print("figures written to", F)
    print(sorted(os.listdir(F)))

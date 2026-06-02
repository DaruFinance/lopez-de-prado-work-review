#!/usr/bin/env python3
"""
uniqueness_deflation.py: link sample uniqueness (Ch.4) to the multiple-testing
deflation penalty (Ch.8).

LIGHT, recompute-only. Reads the banked per-instrument / by-market tables, takes
the effective sample size that the uniqueness accounting already produced
(effective_N = sum of label uniqueness ~= eff_ratio * N), and shows what that
honest sample size does to the Sharpe-significance math: the Probabilistic and
Deflated Sharpe Ratios and the Minimum Track Record Length all depend on the
SAMPLE SIZE, so feeding the nominal N (treating ~0.4*N redundant observations as
independent) overstates statistical significance.

No model training, no large data load. Reuses lib/overfit.py unchanged.

Outputs:
  tables/uniqueness_vs_deflation.csv
  tables/uniqueness_vs_deflation.md
  figures/fig_uniqueness_deflation.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, os.path.join(ROOT, "lib"))

import overfit as OF
import style as ST
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(HERE)
TBL = os.path.join(PROJ, "tables")
FIG = os.path.join(PROJ, "figures")

# --------------------------------------------------------------------------- #
# Assumptions (stated). The banked study is a classification study (log-loss /
# AUC); it does not bank a per-strategy Sharpe ratio. To make the deflation
# point concrete we evaluate the math at a single REPRESENTATIVE per-observation
# (per-label) out-of-sample Sharpe and a single representative selection trial
# count, held FIXED across markets so the only thing that moves the deflation
# numbers is the sample size N vs effective_N. These are illustrative inputs to
# the formulas, not measured returns.
# --------------------------------------------------------------------------- #
SR_OBS = 0.025         # per-label OOS Sharpe (modest single-series edge),
                       # chosen in the sensitive band where the sample size
                       # materially moves the significance verdict
N_TRIALS = 100         # selection trials behind the chosen strategy
VAR_SR_TRIALS = 6.4e-5 # dispersion of trial Sharpes (var of per-obs SR
                       # estimates); gives an expected-max benchmark sr0 ~ 0.020,
                       # just below SR_OBS so the DSR test is informative
SKEW = 0.0             # Gaussian return assumption (stated)
KURT = 3.0             # non-excess kurtosis (Gaussian)
PSR_BENCH = 0.0        # PSR vs a zero-Sharpe benchmark
TRL_PROB = 0.95        # confidence for MinTRL


def sharpe_se(sr: float, n_obs: float, skew: float, kurt: float) -> float:
    """Standard error of a Sharpe estimate (Lo 2002 / Bailey-LdP variance term).
    Var(SR_hat) = (1 - skew*SR + (kurt-1)/4 * SR^2) / (n - 1)."""
    if n_obs < 2:
        return np.nan
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n_obs - 1.0)
    return float(np.sqrt(var))


# Expected-max-Sharpe benchmark from N_TRIALS skill-less trials with the stated
# dispersion. This is the DSR null and is the SAME for both columns; only the
# sample size n_obs differs, so the DSR gap is driven purely by the honest count.
SR0 = OF.expected_max_sharpe(N_TRIALS, VAR_SR_TRIALS)


def deflated(sr, n_obs):
    """DSR: PSR of the strategy against the expected-max-Sharpe benchmark SR0,
    evaluated at sample size n_obs. Reuses the library PSR verbatim."""
    dsr = OF.prob_sharpe_ratio(sr, int(round(n_obs)), SKEW, KURT, sr_benchmark=SR0)
    return dsr, SR0


def row_for(market, N, eff_ratio):
    effN = N * eff_ratio
    psr_nom = OF.prob_sharpe_ratio(SR_OBS, int(round(N)), SKEW, KURT, PSR_BENCH)
    psr_eff = OF.prob_sharpe_ratio(SR_OBS, int(round(effN)), SKEW, KURT, PSR_BENCH)
    se_nom = sharpe_se(SR_OBS, N, SKEW, KURT)
    se_eff = sharpe_se(SR_OBS, effN, SKEW, KURT)
    dsr_nom, sr0 = deflated(SR_OBS, N)
    dsr_eff, _ = deflated(SR_OBS, effN)
    trl_nom = OF.min_track_record_length(SR_OBS, SKEW, KURT, PSR_BENCH, TRL_PROB)
    trl_eff = trl_nom  # MinTRL is a REQUIRED count, independent of N; see note
    # significance-inflation factor: how much the PSR z-stat is overstated by
    # using N instead of effective_N. z scales with sqrt(n_obs - 1), so the
    # factor is sqrt((N-1)/(effN-1)) ~= 1/sqrt(eff_ratio).
    z_infl = np.sqrt((N - 1.0) / (effN - 1.0))
    se_infl = se_eff / se_nom
    return dict(
        market=market, N=int(round(N)), effective_N=int(round(effN)),
        eff_ratio=round(eff_ratio, 4),
        sharpe_se_nominal=round(se_nom, 5),
        sharpe_se_effective=round(se_eff, 5),
        se_inflation=round(se_infl, 3),
        psr_nominal=round(psr_nom, 4),
        psr_effective=round(psr_eff, 4),
        psr_drop=round(psr_nom - psr_eff, 4),
        dsr_nominal=round(dsr_nom, 4),
        dsr_effective=round(dsr_eff, 4),
        dsr_drop=round(dsr_nom - dsr_eff, 4),
        z_inflation_factor=round(z_infl, 3),
        min_trl=round(trl_nom, 1),
    )


def main():
    bm = pd.read_csv(os.path.join(TBL, "by_market_summary.csv"))
    rows = []
    for _, r in bm.iterrows():
        rows.append(row_for(r["market"], float(r["med_N"]), float(r["med_eff_ratio"])))
    # pooled row across all instruments (median N and median eff_ratio over all)
    pi = pd.read_csv(os.path.join(TBL, "per_instrument.csv"))
    rows.append(row_for("all", float(pi["N"].median()), float(pi["eff_ratio"].median())))
    df = pd.DataFrame(rows)

    os.makedirs(TBL, exist_ok=True)
    csv_path = os.path.join(TBL, "uniqueness_vs_deflation.csv")
    df.to_csv(csv_path, index=False)

    # markdown
    md = ["# Sample uniqueness vs the deflation penalty",
          "",
          "Per-observation OOS Sharpe held fixed at "
          f"`SR = {SR_OBS}`; selection trials `N_trials = {N_TRIALS}`, "
          f"trial-Sharpe variance `{VAR_SR_TRIALS}`, Gaussian returns "
          f"(skew {SKEW}, kurtosis {KURT}). Only the sample size moves between "
          "the nominal and effective columns.",
          "",
          "PSR is the probability the true Sharpe beats zero. DSR is the PSR "
          "against the expected-max-Sharpe benchmark implied by the trials. The "
          "z-inflation factor is how much the PSR test statistic is overstated by "
          "using the nominal row count instead of the honest effective count.",
          "",
          "| market | N | effective N | eff ratio | Sharpe SE (nominal) | Sharpe SE (effective) | SE inflation | PSR (nominal) | PSR (effective) | PSR drop | DSR (nominal) | DSR (effective) | DSR drop | z-inflation |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append("| {market} | {N:,} | {effN:,} | {er:.3f} | {sen:.5f} | {see:.5f} | "
                   "{sei:.3f} | {pn:.4f} | {pe:.4f} | {pd:.4f} | {dn:.4f} | {de:.4f} | "
                   "{dd:.4f} | {zi:.3f} |".format(
                       market=r["market"], N=r["N"], effN=r["effective_N"],
                       er=r["eff_ratio"], sen=r["sharpe_se_nominal"],
                       see=r["sharpe_se_effective"], sei=r["se_inflation"],
                       pn=r["psr_nominal"], pe=r["psr_effective"], pd=r["psr_drop"],
                       dn=r["dsr_nominal"], de=r["dsr_effective"], dd=r["dsr_drop"],
                       zi=r["z_inflation_factor"]))
    md += ["",
           f"Minimum track record length to reach {int(TRL_PROB*100)} percent "
           f"PSR at this Sharpe is {df['min_trl'].iloc[0]:.0f} independent "
           "observations. It is a required count of independent observations, so "
           "it does not change with the sample you happen to hold. The point is "
           "the comparison: a nominal sample of about 15,000 overlapping labels "
           "supplies only about 6,000 to 6,700 independent observations, so a "
           "track that looks long enough on a row count can fall short on an "
           "honest count.",
           ""]
    md_path = os.path.join(TBL, "uniqueness_vs_deflation.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    # figure: PSR nominal vs effective (left) and z-inflation factor (right)
    ST.set_style()
    plot = df[df["market"] != "all"].copy()
    markets = plot["market"].tolist()
    x = np.arange(len(markets))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    w = 0.38
    ax1.bar(x - w / 2, plot["dsr_nominal"], w, label="nominal N",
            color=ST.PALETTE["time"])
    ax1.bar(x + w / 2, plot["dsr_effective"], w, label="effective N",
            color=ST.PALETTE["dollar"])
    ax1.set_xticks(x); ax1.set_xticklabels(markets)
    ax1.set_ylabel("Deflated Sharpe Ratio")
    ax1.set_title(f"DSR falls on the honest sample size (SR = {SR_OBS})")
    ax1.set_ylim(0, 0.85)
    for xi, vn, ve in zip(x, plot["dsr_nominal"], plot["dsr_effective"]):
        ax1.text(xi - w / 2, vn + 0.01, f"{vn:.2f}", ha="center", va="bottom", fontsize=9)
        ax1.text(xi + w / 2, ve + 0.01, f"{ve:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.legend(loc="lower center", ncol=2)

    ax2.bar(x, plot["z_inflation_factor"], 0.55, color=ST.PALETTE["accent"])
    ax2.axhline(1.0, color="#333333", lw=0.8, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(markets)
    ax2.set_ylabel("significance inflation (z-stat overstatement)")
    ax2.set_title("Using row count overstates significance by ~1.5x")
    for xi, v in zip(x, plot["z_inflation_factor"]):
        ax2.text(xi, v + 0.02, f"{v:.2f}x", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, max(plot["z_inflation_factor"]) * 1.25)

    fig.tight_layout()
    fig_path = os.path.join(FIG, "fig_uniqueness_deflation.png")
    fig.savefig(fig_path)
    plt.close(fig)

    print("wrote:")
    print(" ", csv_path)
    print(" ", md_path)
    print(" ", fig_path)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

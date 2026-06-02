#!/usr/bin/env python3
"""
Project 0 completeness add-on: Minimum Track Record Length (MinTRL) and
Minimum Backtest Length (MinBTL) per market.

LIGHT, recompute-only. No model training, no large data load. Reads the banked
corpus summaries + per-pair tables and reuses lib/overfit.py (unmodified) to
turn the "best of ~50k fails deflation" verdict into a concrete statement of
how long a track record would have to be for the observed best to be credible.

All Sharpe ratios in overfit.py are PER-OBSERVATION. The corpus runs on daily
PnL annualised with sqrt(252), so:
    sr_per_obs = sr_ann / sqrt(252)
    years      = n_observations / 252

MinTRL benchmark = the False-Strategy-Theorem expected-max null (sr0), i.e. the
credibility bar the deflated test actually uses. Skew/kurtosis are taken as the
Gaussian reference (0, 3) to match the corpus-level DSR computation in
run_corpus_overfit.py; this is stated in the writeup.

RAM: trivial (a handful of scalars per market).
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
sys.path.insert(0, cfg.LIB)
import overfit as OF
import style as ST
ST.set_style()

PROJ = os.path.dirname(HERE)
ANN = np.sqrt(252.0)          # daily -> annualised Sharpe (matches the corpus harness)
TRADING_DAYS = 252.0
MARKETS = ["crypto", "equity", "fx"]
LABEL = {"crypto": "Crypto", "equity": "US Equity", "fx": "Forex"}
# Gaussian reference moments, matching the corpus-level DSR in run_corpus_overfit.py
SKEW = 0.0
KURT = 3.0


def load_market(m):
    """Pull the banked corpus-level numbers + the per-pair table for one market."""
    s = pd.read_csv(f"{PROJ}/tables/corpus_summary_{m}.csv").iloc[0].to_dict()
    pp = pd.read_csv(f"{PROJ}/tables/corpus_per_pair_{m}.csv")
    # The corpus best Sharpe is the max over pairs; the winning pair's own track
    # length is the real track length that strategy actually ran.
    win = pp.loc[pp["best_sr_ann"].idxmax()]
    return dict(
        market=m,
        n_strat=int(s["n_strategies"]),
        eff_n_pooled=int(s["eff_n_pooled"]),
        best_sr_ann=float(s["best_sr_ann"]),
        sr0_ann=float(s["sr0_ann"]),            # False-Strategy-Theorem null (annualised)
        dsr=float(s["dsr"]),
        win_pair=str(win["pair"]),
        T_obs=int(win["T"]),                    # observed track length (days) of the winner
        T_median=int(np.median(pp["T"])),       # corpus-representative track length
    )


def per_market_row(d):
    best_obs = d["best_sr_ann"] / ANN           # per-observation best Sharpe
    sr0_obs = d["sr0_ann"] / ANN                # per-observation null benchmark
    T_obs = d["T_obs"]

    # ---- MinTRL: track length needed for PSR(best) > 95% vs the null benchmark.
    # If the observed best is at or below the null, no finite track makes it
    # credible against that bar -> infinite (which IS the verdict).
    mintrl_obs = OF.min_track_record_length(best_obs, SKEW, KURT,
                                            sr_benchmark=sr0_obs, prob=0.95)
    # For context, MinTRL against a zero benchmark (credible the SR is simply > 0).
    mintrl_vs0 = OF.min_track_record_length(best_obs, SKEW, KURT,
                                            sr_benchmark=0.0, prob=0.95)

    # ---- MinBTL: track length so a skill-less search of N trials would NOT be
    # expected to manufacture a Sharpe as high as the one observed.
    minbtl_nom = OF.min_backtest_length(d["n_strat"], best_obs)
    minbtl_eff = OF.min_backtest_length(d["eff_n_pooled"], best_obs)

    def yrs(x):
        return float("inf") if not np.isfinite(x) else x / TRADING_DAYS

    long_enough = np.isfinite(minbtl_nom) and (T_obs >= minbtl_nom)
    long_enough_eff = np.isfinite(minbtl_eff) and (T_obs >= minbtl_eff)

    return dict(
        market=LABEL[d["market"]],
        winning_pair=d["win_pair"],
        n_trials=d["n_strat"],
        eff_n=d["eff_n_pooled"],
        best_SR_ann=round(d["best_sr_ann"], 3),
        null_E_max_ann=round(d["sr0_ann"], 3),
        obs_track_days=T_obs,
        obs_track_years=round(T_obs / TRADING_DAYS, 2),
        MinTRL_days=(np.inf if not np.isfinite(mintrl_obs) else round(mintrl_obs, 1)),
        MinTRL_years=(np.inf if not np.isfinite(mintrl_obs) else round(yrs(mintrl_obs), 2)),
        MinTRL_vs0_years=round(yrs(mintrl_vs0), 2),
        MinBTL_nominalN_days=(np.nan if not np.isfinite(minbtl_nom) else round(minbtl_nom, 1)),
        MinBTL_nominalN_years=(np.nan if not np.isfinite(minbtl_nom) else round(yrs(minbtl_nom), 2)),
        MinBTL_effN_days=(np.nan if not np.isfinite(minbtl_eff) else round(minbtl_eff, 1)),
        MinBTL_effN_years=(np.nan if not np.isfinite(minbtl_eff) else round(yrs(minbtl_eff), 2)),
        track_clears_MinBTL_nominal=("yes" if long_enough else "no"),
        track_clears_MinBTL_eff=("yes" if long_enough_eff else "no"),
        # verdict on the credibility bar: best must beat the null at all
        best_above_null=("yes" if best_obs > sr0_obs else "no"),
    )


def main():
    rows = [per_market_row(load_market(m)) for m in MARKETS]
    df = pd.DataFrame(rows)

    df.to_csv(f"{PROJ}/tables/min_track_record_length.csv", index=False)

    # human-readable markdown (compact, the load-bearing columns)
    show = df[["market", "winning_pair", "n_trials", "eff_n", "best_SR_ann",
               "null_E_max_ann", "obs_track_years", "MinTRL_years",
               "MinBTL_nominalN_years", "MinBTL_effN_years",
               "track_clears_MinBTL_nominal", "best_above_null"]].copy()
    show.columns = ["market", "winning pair", "N trials", "eff-N",
                    "best SR (ann)", "null E[max] (ann)", "observed track (yrs)",
                    "MinTRL (yrs)", "MinBTL nominal-N (yrs)", "MinBTL eff-N (yrs)",
                    "track >= MinBTL?", "best > null?"]
    with open(f"{PROJ}/tables/min_track_record_length.md", "w") as fh:
        fh.write("# Minimum Track Record Length (MinTRL) and Minimum Backtest "
                 "Length (MinBTL) per market\n\n")
        fh.write("MinTRL is computed against the multiple-testing null benchmark "
                 "(the False-Strategy-Theorem expected maximum), the same bar the "
                 "deflated test uses. MinBTL is the track length so a skill-less "
                 "search of N trials would not be expected to produce a Sharpe as "
                 "high as the one observed. Sharpe ratios are annualised for "
                 "display; the underlying statistics are per observation (daily). "
                 "Skew and kurtosis use the Gaussian reference (0, 3), matching the "
                 "corpus-level deflated-Sharpe computation.\n\n")
        fh.write(show.to_markdown(index=False))
        fh.write("\n\nInfinite MinTRL means the observed best Sharpe is at or "
                 "below the multiple-testing null, so no finite track record makes "
                 "it credible against that benchmark.\n")

    print(show.to_string(index=False))

    # ---- figure: observed track vs required (MinBTL) per market, log scale ----
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(MARKETS))
    w = 0.26
    obs = df["obs_track_years"].to_numpy(float)
    btl_nom = df["MinBTL_nominalN_years"].to_numpy(float)
    btl_eff = df["MinBTL_effN_years"].to_numpy(float)

    b1 = ax.bar(x - w, obs, w, label="observed track length",
                color=ST.PALETTE["dollar"])
    b2 = ax.bar(x, btl_nom, w, label="MinBTL (nominal N trials)",
                color=ST.PALETTE["accent"])
    b3 = ax.bar(x + w, btl_eff, w, label="MinBTL (effective N trials)",
                color=ST.PALETTE["volume"])
    ax.set_yscale("log")
    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            if np.isfinite(h) and h > 0:
                ax.text(r.get_x() + r.get_width() / 2, h * 1.05, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MARKETS])
    ax.set_ylabel("track length (years, log scale)")
    ax.set_title("Observed track length vs the track length required to trust "
                 "the best of the search")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{PROJ}/figures/fig_min_track_record.png", dpi=130)
    plt.close(fig)

    print("\nwrote tables/min_track_record_length.{csv,md} and "
          "figures/fig_min_track_record.png")
    return df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Project 1 — Information-Driven Bars (López de Prado, AFML Ch. 2), at scale.

Reproduces LdP's claim that information-driven bars (especially DOLLAR bars)
yield returns closer to IID-Gaussian and counts more stable through time than
fixed TIME bars — and tests it across the full liquid Binance USD-M perp
cross-section (real 30m data with trade count + taker-buy volume).

Outputs:
  tables/per_pair_return_stats.csv     one row per (pair, bar_type)
  tables/summary_by_bartype.csv        cross-sectional aggregate
  tables/summary_by_bartype.md         same, markdown
  figures/*.png                        publication figures
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

sys.path.insert(0, "/home/daru/ldp_review/lib")
import bars as B
import barstats as S
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/01_information_driven_bars"
DATA = sorted(glob.glob("/mnt/d/T5_StatArb_Data/binance_perp/*_30m.parquet"))
BAR_TYPES = ["time", "tick", "volume", "dollar"]
BASE_PER_DAY = 48            # 48 x 30m = 1 day -> target ~daily-frequency bars
MIN_BASE = 5000              # require enough history (~3.5 months of 30m)


def pair_name(path):
    return os.path.basename(path).replace("_30m.parquet", "")


def process_pair(path):
    name = pair_name(path)
    try:
        df = B.load_base(path)
    except Exception as e:
        return None, None, f"{name}: load failed {e}"
    if len(df) < MIN_BASE:
        return None, None, f"{name}: too short ({len(df)})"

    n_target = max(50, len(df) // BASE_PER_DAY)     # ~daily bar count
    barset = B.matched_bars(df, n_target)

    rows, stab = [], []
    for bt in BAR_TYPES:
        bars = barset[bt]
        r = B.log_returns(bars)
        st = S.return_stats(r)
        cs = S.count_stability(bars, freq="W")
        rows.append(dict(pair=name, bar_type=bt, n_bars=len(bars),
                         **{k: st[k] for k in ["n", "skew", "exkurt", "jb",
                                               "jb_p", "ac1", "lb_p"]},
                         count_cv=cs["cv"]))
        stab.append((bt, st.get("std_returns"), cs.get("series")))
    return pd.DataFrame(rows), {bt: (sr, sc) for bt, sr, sc in stab}, name


def main():
    print(f"Loading {len(DATA)} candidate pairs ...")
    per_pair, examples = [], {}
    used = []
    for p in DATA:
        res, ex, msg = process_pair(p)
        if res is None:
            print("  skip", msg); continue
        per_pair.append(res); used.append(msg)
        if msg in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            examples[msg] = ex
    print(f"Used {len(used)} pairs: {', '.join(used)}")

    df = pd.concat(per_pair, ignore_index=True)
    df.to_csv(f"{PROJ}/tables/per_pair_return_stats.csv", index=False)

    # ---- cross-sectional summary -------------------------------------------
    agg = (df.assign(abs_skew=df["skew"].abs(), abs_ac1=df["ac1"].abs())
             .groupby("bar_type")
             .agg(pairs=("pair", "nunique"),
                  med_abs_skew=("abs_skew", "median"),
                  med_exkurt=("exkurt", "median"),
                  med_abs_ac1=("abs_ac1", "median"),
                  med_jb=("jb", "median"),
                  frac_normal=("jb_p", lambda s: float((s > 0.05).mean())),
                  med_count_cv=("count_cv", "median"))
             .reindex(BAR_TYPES))
    agg.to_csv(f"{PROJ}/tables/summary_by_bartype.csv")
    with open(f"{PROJ}/tables/summary_by_bartype.md", "w") as fh:
        fh.write("# Cross-sectional summary by bar type (median across pairs)\n\n")
        fh.write("Lower |skew|, lower excess kurtosis, lower |AC(1)|, lower JB, "
                 "lower count-CV are better; frac_normal = share of pairs whose "
                 "returns fail to reject normality at 5% (higher=more Gaussian).\n\n")
        fh.write(agg.round(4).to_markdown())
    print("\n=== SUMMARY BY BAR TYPE (median across pairs) ===")
    print(agg.round(4).to_string())

    make_figures(df, agg, examples)
    print("\nDone. Tables + figures written under", PROJ)


def make_figures(df, agg, examples):
    fig_dir = f"{PROJ}/figures"

    # Fig 1 — excess kurtosis distribution by bar type (the headline)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [df.loc[df.bar_type == bt, "exkurt"].dropna() for bt in BAR_TYPES]
    bp = ax.boxplot(data, labels=BAR_TYPES, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, bt in zip(bp["boxes"], BAR_TYPES):
        patch.set_facecolor(ST.barcolor(bt)); patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Excess kurtosis of bar returns")
    ax.set_title("Returns are closer to Gaussian under information-driven bars\n"
                 "(excess kurtosis across the perp cross-section; 0 = Gaussian)")
    ax.text(0.99, 0.97, "Gaussian = 0", transform=ax.transAxes, ha="right",
            va="top", fontsize=9, color="gray")
    fig.savefig(f"{fig_dir}/fig1_excess_kurtosis.png"); plt.close(fig)

    # Fig 2 — |skew| and |AC(1)| panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, ttl in [(axes[0], "skew", "|Skewness|"),
                         (axes[1], "ac1", "|First-order autocorrelation|")]:
        data = [df.loc[df.bar_type == bt, col].abs().dropna() for bt in BAR_TYPES]
        bp = ax.boxplot(data, labels=BAR_TYPES, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch, bt in zip(bp["boxes"], BAR_TYPES):
            patch.set_facecolor(ST.barcolor(bt)); patch.set_alpha(0.75)
        ax.set_title(ttl)
    fig.suptitle("Lower skew and weaker serial correlation under information bars",
                 fontweight="bold")
    fig.savefig(f"{fig_dir}/fig2_skew_autocorr.png"); plt.close(fig)

    # Fig 3 — bar-count stability (coefficient of variation of weekly counts)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [df.loc[df.bar_type == bt, "count_cv"].dropna() for bt in BAR_TYPES]
    bp = ax.boxplot(data, labels=BAR_TYPES, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, bt in zip(bp["boxes"], BAR_TYPES):
        patch.set_facecolor(ST.barcolor(bt)); patch.set_alpha(0.75)
    ax.set_ylabel("CV of weekly bar count (lower = more stable)")
    ax.set_title("Dollar-bar production is the most stable through time")
    fig.savefig(f"{fig_dir}/fig3_count_stability.png"); plt.close(fig)

    # Fig 4 — QQ plot of standardized returns for a representative pair
    name = "BTCUSDT" if "BTCUSDT" in examples else next(iter(examples), None)
    if name and examples.get(name):
        fig, ax = plt.subplots(figsize=(6, 6))
        for bt in BAR_TYPES:
            sr = examples[name][bt][0]
            if sr is None or len(sr) < 50:
                continue
            sr = np.sort(sr.to_numpy())
            q = ss.norm.ppf((np.arange(1, len(sr) + 1) - 0.5) / len(sr))
            ax.plot(q, sr, ".", ms=2.5, color=ST.barcolor(bt), label=bt, alpha=0.7)
        lim = 5
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.9, alpha=0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-8, 8)
        ax.set_xlabel("Theoretical Gaussian quantiles")
        ax.set_ylabel("Standardized return quantiles")
        ax.set_title(f"Normal QQ plot — {name}\n(closer to the dashed line = more Gaussian)")
        ax.legend(markerscale=4)
        fig.savefig(f"{fig_dir}/fig4_qq_{name}.png"); plt.close(fig)

    print("Figures written:", os.listdir(fig_dir))


if __name__ == "__main__":
    main()

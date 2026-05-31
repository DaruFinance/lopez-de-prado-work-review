#!/usr/bin/env python3
"""
Project 1 (clean) — Information-Driven Bars across the cross-section, built from
CLEAN 1m Binance USD-M perp dumps (2022-2024). Replaces the earlier run that
used a contaminated legacy 30m dataset (see writeup §data-quality caveat).

Matched ~daily bars are built directly from 1m base (finest available), so the
information-bar comparison is at the granularity where LdP's method is meant to
operate. Outputs the canonical tables + figures for the project.
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss
from multiprocessing import Pool

sys.path.insert(0, "/home/daru/ldp_review/lib")
import bars as B
import barstats as S
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/01_information_driven_bars"
CACHE = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
BAR_TYPES = ["time", "tick", "volume", "dollar"]


def load_1m(path):
    df = pd.read_parquet(path)
    df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns=["open_time"])
    return df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]


def process(path):
    name = os.path.basename(path).replace("_1m.parquet", "")
    try:
        df = load_1m(path)
    except Exception as e:
        return None
    days = max(50, int((df.index[-1] - df.index[0]).days))
    if len(df) < 50000:
        return None
    barset = B.matched_bars(df, n_target=days)     # ~daily bars from 1m base
    rows = []
    qq = {}
    for bt in BAR_TYPES:
        bars = barset[bt]
        st = S.return_stats(B.log_returns(bars))
        cs = S.count_stability(bars, "W")
        rows.append(dict(pair=name, bar_type=bt, n_bars=len(bars),
                         **{k: st[k] for k in ["n", "skew", "exkurt", "jb",
                                               "jb_p", "ac1", "lb_p"]},
                         count_cv=cs["cv"]))
        qq[bt] = st.get("std_returns")
    return pd.DataFrame(rows), (name, qq)


def main():
    files = sorted(glob.glob(f"{CACHE}/*_1m.parquet"))
    print(f"{len(files)} clean 1m pairs")
    with Pool(min(12, len(files))) as pool:
        out = [r for r in pool.map(process, files) if r is not None]
    per_pair = pd.concat([o[0] for o in out], ignore_index=True)
    qqs = {n: q for n, q in [o[1] for o in out]}
    per_pair.to_csv(f"{PROJ}/tables/per_pair_clean.csv", index=False)
    print(f"used {per_pair.pair.nunique()} pairs")

    agg = (per_pair.assign(abs_skew=per_pair["skew"].abs(),
                           abs_ac1=per_pair["ac1"].abs())
           .groupby("bar_type")
           .agg(pairs=("pair", "nunique"),
                med_abs_skew=("abs_skew", "median"),
                med_exkurt=("exkurt", "median"),
                med_abs_ac1=("abs_ac1", "median"),
                med_jb=("jb", "median"),
                frac_normal=("jb_p", lambda s: float((s > 0.05).mean())),
                med_count_cv=("count_cv", "median"))
           .reindex(BAR_TYPES))
    agg.to_csv(f"{PROJ}/tables/summary_clean.csv")
    with open(f"{PROJ}/tables/summary_clean.md", "w") as fh:
        fh.write("# Information-driven bars — clean 1m cross-section "
                 f"({per_pair.pair.nunique()} Binance perps, 2022-2024, ~daily bars)\n\n")
        fh.write("Median across pairs. Lower |skew|, excess kurtosis, |AC(1)|, JB, "
                 "count-CV are better.\n\n")
        fh.write(agg.round(4).to_markdown())
        # count stability among information bars only
        info = (per_pair[per_pair.bar_type != "time"]
                .groupby("bar_type")["count_cv"].median().reindex(["tick", "volume", "dollar"]))
        fh.write("\n\n## Count stability among information bars (median CV)\n\n")
        fh.write(info.round(4).to_markdown())
    print("\n=== CLEAN SUMMARY (median across pairs) ===")
    print(agg.round(4).to_string())

    figs(per_pair, qqs)
    print("\nClean cross-section done.")


def boxpanel(ax, per_pair, col, absval, title):
    data = [(per_pair.loc[per_pair.bar_type == bt, col].abs() if absval
             else per_pair.loc[per_pair.bar_type == bt, col]).dropna()
            for bt in BAR_TYPES]
    bp = ax.boxplot(data, labels=BAR_TYPES, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, bt in zip(bp["boxes"], BAR_TYPES):
        patch.set_facecolor(ST.barcolor(bt)); patch.set_alpha(0.78)
    ax.set_title(title)


def figs(per_pair, qqs):
    d = f"{PROJ}/figures"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    boxpanel(ax, per_pair, "exkurt", False, "")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Excess kurtosis of bar returns")
    ax.set_title("Information-driven bars Gaussianize returns (clean 1m, "
                 f"{per_pair.pair.nunique()} perps)\n0 = Gaussian; lower is better")
    fig.savefig(f"{d}/fig1_excess_kurtosis_clean.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    boxpanel(axes[0], per_pair, "skew", True, "|Skewness|")
    boxpanel(axes[1], per_pair, "ac1", True, "|First-order autocorrelation|")
    fig.suptitle("Lower skew and weaker serial correlation under information bars (clean 1m)",
                 fontweight="bold")
    fig.savefig(f"{d}/fig2_skew_autocorr_clean.png"); plt.close(fig)

    # count stability among info bars
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    info_types = ["tick", "volume", "dollar"]
    data = [per_pair.loc[per_pair.bar_type == bt, "count_cv"].dropna() for bt in info_types]
    bp = ax.boxplot(data, labels=info_types, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, bt in zip(bp["boxes"], info_types):
        patch.set_facecolor(ST.barcolor(bt)); patch.set_alpha(0.78)
    ax.set_ylabel("CV of weekly bar count")
    ax.set_title("Bar-count stability among information bars (clean 1m)")
    fig.savefig(f"{d}/fig3_count_stability_clean.png"); plt.close(fig)

    name = "BTCUSDT" if "BTCUSDT" in qqs else next(iter(qqs))
    fig, ax = plt.subplots(figsize=(6, 6))
    for bt in BAR_TYPES:
        sr = qqs[name][bt]
        if sr is None or len(sr) < 50:
            continue
        sr = np.sort(sr.to_numpy())
        q = ss.norm.ppf((np.arange(1, len(sr) + 1) - 0.5) / len(sr))
        ax.plot(q, sr, ".", ms=2.5, color=ST.barcolor(bt), label=bt, alpha=0.7)
    ax.plot([-5, 5], [-5, 5], "k--", lw=0.9, alpha=0.7)
    ax.set_xlim(-5, 5); ax.set_ylim(-8, 8)
    ax.set_xlabel("Theoretical Gaussian quantiles")
    ax.set_ylabel("Standardized return quantiles")
    ax.set_title(f"Normal QQ plot — {name} (clean 1m)\ncloser to dashed line = more Gaussian")
    ax.legend(markerscale=4)
    fig.savefig(f"{d}/fig4_qq_{name}_clean.png"); plt.close(fig)
    print("clean figures written")


if __name__ == "__main__":
    main()

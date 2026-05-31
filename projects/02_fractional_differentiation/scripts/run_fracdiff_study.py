#!/usr/bin/env python3
"""
Project 2 — Fractional Differentiation (López de Prado, AFML Ch. 5), at scale.

Reproduces LdP's claim that Fixed-Width Window Fractional Differentiation (FFD)
of log prices yields a STATIONARY series (passes ADF) at a fractional exponent
d* usually well below 1, while still preserving memory (high correlation with
the original price level) — unlike integer differencing (returns, d=1) which is
stationary but erases the level information.

Pipeline:
  1. BTC single-series reproduction (ADF stat & memory vs d)            -> fig1, fig4
  2. Cross-section over the 568 Binance USD-M 1h perps (min-d* search)  -> per_pair csv
  3. Memory-vs-stationarity frontier                                    -> fig3
  4. Value-add: memory destroyed by d=1 vs d*, tau sensitivity          -> fig5, fig6
  5. Distribution of d*                                                 -> fig2

All transforms are causal (no lookahead).  Run: python3 run_fracdiff_study.py
"""
import sys, glob, os, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/daru/ldp_review/lib")
import fracdiff as F
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/02_fractional_differentiation"
FIG = os.path.join(PROJ, "figures")
TAB = os.path.join(PROJ, "tables")
DATA = sorted(glob.glob("/home/daru/crypto_ohlcv_perp_all_1h/binance_um/*_1h.parquet"))
ONEM = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"

MIN_ROWS = 5000
TAU = 1e-5
D_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
N_PROC = 32

os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def pair_name(path):
    return os.path.basename(path).replace("_1h.parquet", "")


def load_logclose(path):
    """Return log(close) as a float array, cleaned of zeros and delisting tail."""
    df = pd.read_parquet(path, columns=["close"])
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy()
    c = c[np.isfinite(c)]
    c = c[c > 0]
    # drop a flat delisting tail (repeated identical closes at the very end)
    if len(c) > 10:
        i = len(c) - 1
        while i > 0 and c[i] == c[i - 1]:
            i -= 1
        c = c[: i + 1]
    return np.log(c)


# --------------------------------------------------------------------------- #
# cross-section worker
# --------------------------------------------------------------------------- #
def process_pair(path):
    name = pair_name(path)
    try:
        x = load_logclose(path)
    except Exception as e:
        return {"pair": name, "error": f"load {e}"}
    if len(x) < MIN_ROWS:
        return {"pair": name, "error": f"short {len(x)}"}

    res = F.min_d_search(x, d_grid=D_GRID, tau=TAU)
    star = res["star"]
    trace = {r["d"]: r for r in res["trace"]}
    r1 = trace.get(1.0, {})
    if star is None:
        return {"pair": name, "n": len(x), "d_star": np.nan,
                "adf_stat_at_dstar": np.nan, "adf_p_at_dstar": np.nan,
                "corr_level_at_dstar": np.nan,
                "corr_level_at_d1": r1.get("corr_level", np.nan),
                "window_len": np.nan, "error": "no_dstar"}
    return {
        "pair": name, "n": len(x),
        "d_star": res["d_star"],
        "adf_stat_at_dstar": star["adf_stat"],
        "adf_p_at_dstar": star["adf_p"],
        "corr_level_at_dstar": star["corr_level"],
        "corr_level_at_d1": r1.get("corr_level", np.nan),
        "window_len": star["window_len"],
        "error": "",
    }


# --------------------------------------------------------------------------- #
# experiment 1 — BTC single series reproduction
# --------------------------------------------------------------------------- #
def btc_reproduction():
    btc = os.path.join(os.path.dirname(DATA[0]), "BTCUSDT_1h.parquet")
    x = load_logclose(btc)
    res = F.min_d_search(x, d_grid=D_GRID, tau=TAU)
    tr = pd.DataFrame(res["trace"])
    tr.to_csv(os.path.join(TAB, "btc_adf_vs_d.csv"), index=False)
    d_star = res["d_star"]
    crit = float(np.nanmedian(tr["adf_crit_5"]))

    # fig1: ADF stat vs d (left) + corr-with-level (right)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(tr["d"], tr["adf_stat"], "-o", color=ST.PALETTE["dollar"],
             ms=4, label="ADF statistic")
    ax1.axhline(crit, ls="--", color=ST.PALETTE["accent"], lw=1.3,
                label=f"95% ADF crit ({crit:.2f})")
    if not np.isnan(d_star):
        ax1.axvline(d_star, ls=":", color="#444", lw=1.2)
        ax1.annotate(f"d* = {d_star:.2f}", xy=(d_star, crit),
                     xytext=(d_star + 0.06, crit - 4),
                     color="#444", fontweight="bold")
    ax1.set_xlabel("fractional exponent  d")
    ax1.set_ylabel("ADF test statistic", color=ST.PALETTE["dollar"])
    ax1.tick_params(axis="y", labelcolor=ST.PALETTE["dollar"])

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(tr["d"], tr["corr_level"], "-s", color=ST.PALETTE["tick"],
             ms=3.5, label="corr with log-price level")
    ax2.set_ylabel("corr( FFD , log-price )", color=ST.PALETTE["tick"])
    ax2.tick_params(axis="y", labelcolor=ST.PALETTE["tick"])
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9)
    ax1.set_title("BTCUSDT 1h — stationarity (ADF) vs memory across d  (FFD, tau=1e-5)")
    fig.savefig(os.path.join(FIG, "fig1_adf_vs_d_BTC.png"))
    plt.close(fig)

    # fig4: BTC log-price vs its FFD(d*) overlay
    y = F.ffd(x, d_star, tau=TAU)
    fig, axA = plt.subplots(figsize=(7.6, 4.2))
    axA.plot(x, color=ST.PALETTE["time"], lw=0.8, label="log close (d=0, level)")
    axA.set_ylabel("log price", color=ST.PALETTE["time"])
    axA.tick_params(axis="y", labelcolor=ST.PALETTE["time"])
    axA.set_xlabel("hours since listing")
    axB = axA.twinx()
    axB.spines["top"].set_visible(False)
    axB.plot(y, color=ST.PALETTE["dollar"], lw=0.6,
             label=f"FFD(d*={d_star:.2f}) — stationary")
    axB.set_ylabel(f"FFD(d*={d_star:.2f})", color=ST.PALETTE["dollar"])
    axB.tick_params(axis="y", labelcolor=ST.PALETTE["dollar"])
    axB.grid(False)
    lA, labA = axA.get_legend_handles_labels()
    lB, labB = axB.get_legend_handles_labels()
    axA.legend(lA + lB, labA + labB, loc="upper left", fontsize=9)
    star_row = res["star"]
    axA.set_title(f"BTCUSDT — level vs FFD(d*)  (ADF={star_row['adf_stat']:.2f}, "
                  f"corr w/ level={star_row['corr_level']:.2f})")
    fig.savefig(os.path.join(FIG, "fig4_btc_level_vs_ffd.png"))
    plt.close(fig)

    return res, tr, crit


# --------------------------------------------------------------------------- #
# experiment 4b — tau sensitivity on BTC
# --------------------------------------------------------------------------- #
def tau_sensitivity():
    btc = os.path.join(os.path.dirname(DATA[0]), "BTCUSDT_1h.parquet")
    x = load_logclose(btc)
    taus = [1e-3, 1e-4, 1e-5]
    rows = []
    for tau in taus:
        res = F.min_d_search(x, d_grid=D_GRID, tau=tau)
        star = res["star"]
        rows.append(dict(tau=tau, d_star=res["d_star"],
                         window_len_at_dstar=(star["window_len"] if star else np.nan),
                         adf_at_dstar=(star["adf_stat"] if star else np.nan),
                         corr_level_at_dstar=(star["corr_level"] if star else np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "btc_tau_sensitivity.csv"), index=False)

    # fig6: window length vs d for each tau
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    cols = [ST.PALETTE["time"], ST.PALETTE["tick"], ST.PALETTE["dollar"]]
    for tau, col in zip(taus, cols):
        wl = [len(F.ffd_weights(d, tau)) for d in D_GRID]
        ax.plot(D_GRID, wl, "-o", ms=3, color=col, label=f"tau={tau:g}")
    ax.set_yscale("log")
    ax.set_xlabel("fractional exponent  d")
    ax.set_ylabel("FFD window length  W  (log scale)")
    ax.set_title("Truncation tau sets the FFD window length (BTC log-price)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig6_tau_window_length.png"))
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
# experiment 2/3 — cross-section
# --------------------------------------------------------------------------- #
def cross_section():
    with Pool(N_PROC) as pool:
        results = pool.map(process_pair, DATA, chunksize=4)
    df = pd.DataFrame(results)
    ok = df[(df["error"] == "") & df["d_star"].notna()].copy()
    df.to_csv(os.path.join(TAB, "per_pair_fracdiff_raw.csv"), index=False)
    keep = ["pair", "n", "d_star", "adf_stat_at_dstar", "adf_p_at_dstar",
            "corr_level_at_dstar", "corr_level_at_d1", "window_len"]
    ok[keep].sort_values("d_star").to_csv(
        os.path.join(TAB, "per_pair_fracdiff.csv"), index=False)
    return df, ok


def frontier_panel():
    """Build the memory-vs-d frontier across the cross-section (median + IQR)."""
    rows = []
    with Pool(N_PROC) as pool:
        allres = pool.map(_frontier_worker, DATA, chunksize=4)
    for r in allres:
        if r:
            rows.extend(r)
    fr = pd.DataFrame(rows, columns=["d", "corr", "passed"])
    fr.to_csv(os.path.join(TAB, "frontier_long.csv"), index=False)
    return fr


def _frontier_worker(path):
    try:
        x = load_logclose(path)
    except Exception:
        return None
    if len(x) < MIN_ROWS:
        return None
    from statsmodels.tsa.stattools import adfuller
    out = []
    for d in D_GRID:
        w = F.ffd_weights(d, TAU)
        y = F.ffd(x, d, tau=TAU, weights=w)
        m = ~np.isnan(y)
        if m.sum() > 50 and np.std(y[m]) > 0:
            c = float(np.corrcoef(y[m], x[m])[0, 1])
            try:
                r = adfuller(y[m], maxlag=1, regression="c", autolag=None)
                p = bool(r[0] < r[4]["5%"])
            except Exception:
                p = False
        else:
            c, p = np.nan, False
        out.append((float(d), c, int(p)))
    return out


# --------------------------------------------------------------------------- #
# figures for cross-section
# --------------------------------------------------------------------------- #
def fig_dstar_distribution(ok):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8),
                                   gridspec_kw={"width_ratios": [3, 1]})
    ds = ok["d_star"].dropna()
    med = ds.median()
    ax1.hist(ds, bins=np.arange(-0.025, 1.05, 0.05),
             color=ST.PALETTE["dollar"], edgecolor="white", alpha=0.9)
    ax1.axvline(med, ls="--", color=ST.PALETTE["accent"], lw=1.5,
                label=f"median d* = {med:.2f}")
    ax1.axvline(1.0, ls=":", color="#888", lw=1.2, label="d=1 (returns)")
    ax1.set_xlabel("minimum d* for stationarity")
    ax1.set_ylabel("number of pairs")
    ax1.legend(fontsize=9)
    ax1.set_title(f"d* across {len(ds)} Binance USD-M perps (1h)")

    ax2.boxplot(ds, vert=True, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=ST.PALETTE["tick"], alpha=0.7),
                medianprops=dict(color=ST.PALETTE["accent"], lw=2))
    ax2.set_xticks([])
    ax2.set_ylabel("d*")
    ax2.set_title("box")
    fig.savefig(os.path.join(FIG, "fig2_dstar_distribution.png"))
    plt.close(fig)
    return med


def fig_frontier(fr, med_dstar):
    g = fr.dropna(subset=["corr"]).groupby("d")["corr"]
    med = g.median()
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    passfrac = fr.groupby("d")["passed"].mean()

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.fill_between(med.index, q1, q3, color=ST.PALETTE["tick"], alpha=0.22,
                    label="IQR across pairs")
    ax.plot(med.index, med.values, "-o", color=ST.PALETTE["tick"], ms=4,
            label="median corr with level")
    ax.axvline(med_dstar, ls="--", color=ST.PALETTE["accent"], lw=1.4,
               label=f"median d* = {med_dstar:.2f}")
    ax.set_xlabel("fractional exponent  d")
    ax.set_ylabel("corr( FFD , log-price level )  — memory retained")
    ax.set_ylim(-0.05, 1.05)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(passfrac.index, passfrac.values, "-^", color=ST.PALETTE["dollar"],
             ms=3.5, label="frac. pairs ADF-stationary")
    ax2.set_ylabel("fraction of pairs ADF-stationary",
                   color=ST.PALETTE["dollar"])
    ax2.tick_params(axis="y", labelcolor=ST.PALETTE["dollar"])
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8.5)
    ax.set_title("Memory-vs-stationarity frontier across the cross-section")
    fig.savefig(os.path.join(FIG, "fig3_memory_frontier.png"))
    plt.close(fig)


def fig_memory_loss(ok):
    """fig5: paired memory retained at d* vs at d=1 (returns)."""
    sub = ok.dropna(subset=["corr_level_at_dstar", "corr_level_at_d1"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    parts = ax.violinplot([sub["corr_level_at_dstar"].values,
                           sub["corr_level_at_d1"].abs().values],
                          showmedians=True, widths=0.8)
    for pc, col in zip(parts["bodies"], [ST.PALETTE["dollar"], ST.PALETTE["time"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#333")
    ax.set_xticks([1, 2])
    m1 = sub["corr_level_at_dstar"].median()
    m2 = sub["corr_level_at_d1"].abs().median()
    ax.set_xticklabels([f"FFD at d*\n(median {m1:.2f})",
                        f"returns d=1\n(median |corr| {m2:.2f})"])
    ax.set_ylabel("|corr( differenced series , log-price level )|")
    ax.set_title("Memory retained: FFD(d*) vs plain returns (d=1)")
    fig.savefig(os.path.join(FIG, "fig5_memory_loss.png"))
    plt.close(fig)
    return m1, m2


# --------------------------------------------------------------------------- #
# value-add: relate d* to volatility / trend across the cross section
# --------------------------------------------------------------------------- #
def _asset_props(path):
    name = pair_name(path)
    try:
        x = load_logclose(path)
    except Exception:
        return None
    if len(x) < MIN_ROWS:
        return None
    r = np.diff(x)
    vol = float(np.std(r))
    # trend strength = |total drift| / (vol * sqrt(n))  (signal-to-noise of path)
    n = len(r)
    trend = float(abs(r.sum()) / (vol * np.sqrt(n) + 1e-12))
    return {"pair": name, "ann_vol_proxy": vol, "trend_strength": trend}


def asset_property_link(ok):
    with Pool(N_PROC) as pool:
        props = [p for p in pool.map(_asset_props, DATA, chunksize=8) if p]
    pdf = pd.DataFrame(props)
    merged = ok.merge(pdf, on="pair", how="inner").dropna(
        subset=["d_star", "ann_vol_proxy", "trend_strength"])
    merged.to_csv(os.path.join(TAB, "dstar_vs_properties.csv"), index=False)

    rho_vol = merged["d_star"].corr(merged["ann_vol_proxy"], method="spearman")
    rho_tr = merged["d_star"].corr(merged["trend_strength"], method="spearman")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    a1.scatter(merged["ann_vol_proxy"], merged["d_star"], s=10,
               color=ST.PALETTE["tick"], alpha=0.5)
    a1.set_xlabel("per-bar return vol (1h)")
    a1.set_ylabel("d*")
    a1.set_title(f"d* vs volatility  (Spearman {rho_vol:+.2f})")
    a2.scatter(merged["trend_strength"], merged["d_star"], s=10,
               color=ST.PALETTE["dollar"], alpha=0.5)
    a2.set_xlabel("trend strength  |drift|/(vol*sqrt n)")
    a2.set_ylabel("d*")
    a2.set_title(f"d* vs trend strength  (Spearman {rho_tr:+.2f})")
    fig.savefig(os.path.join(FIG, "fig7_dstar_vs_props.png"))
    plt.close(fig)
    return rho_vol, rho_tr, merged


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def write_summary(ok, btc_res, med_dstar, mem_dstar, mem_d1, tau_df,
                  rho_vol, rho_tr, crit):
    ds = ok["d_star"].dropna()
    frac_below1 = float((ds < 1.0).mean())
    q = ds.quantile([0.25, 0.5, 0.75])
    btc_star = btc_res["star"]

    lines = []
    lines.append("# Fractional Differentiation — cross-sectional summary\n")
    lines.append(f"- Pairs with a valid d* on the [0,1] grid: **{len(ds)}** "
                 f"(of {len(DATA)} candidate 1h perps; rest too short or no d* found)\n")
    lines.append(f"- **Median d\\* = {q[0.5]:.3f}**  (Q1 {q[0.25]:.3f}, Q3 {q[0.75]:.3f})\n")
    lines.append(f"- Fraction of pairs with d\\* < 1: **{frac_below1:.1%}**\n")
    lines.append(f"- Median memory retained (corr w/ level) at d\\*: **{mem_dstar:.3f}**\n")
    lines.append(f"- Median |memory| at d=1 (plain returns): **{mem_d1:.3f}**  "
                 f"→ returns destroy ~{(1 - mem_d1/max(mem_dstar,1e-9)):.0%} of the "
                 f"memory that FFD(d\\*) keeps\n")
    lines.append(f"- BTCUSDT: d\\* = **{btc_res['d_star']:.2f}**, "
                 f"ADF = {btc_star['adf_stat']:.2f} (95% crit {crit:.2f}), "
                 f"corr w/ level = {btc_star['corr_level']:.3f}, "
                 f"window = {btc_star['window_len']} bars\n")
    lines.append(f"- d\\* vs per-bar vol: Spearman {rho_vol:+.2f};  "
                 f"d\\* vs trend strength: Spearman {rho_tr:+.2f}\n")
    lines.append("\n## tau sensitivity (BTC)\n")
    lines.append(tau_df.to_markdown(index=False))
    lines.append("\n\n## d* quantiles across the cross-section\n")
    lines.append(ds.describe(percentiles=[.05, .25, .5, .75, .95]).to_frame("d_star").to_markdown())
    with open(os.path.join(TAB, "summary_fracdiff.md"), "w") as f:
        f.write("\n".join(str(x) for x in lines))


# --------------------------------------------------------------------------- #
def main():
    print("[1/6] BTC reproduction ...")
    btc_res, btc_tr, crit = btc_reproduction()
    print(f"      BTC d* = {btc_res['d_star']:.2f}")

    print("[2/6] tau sensitivity ...")
    tau_df = tau_sensitivity()
    print(tau_df.to_string(index=False))

    print("[3/6] cross-section min-d* search (568 pairs) ...")
    df, ok = cross_section()
    print(f"      valid d* for {len(ok)} pairs")

    print("[4/6] memory-vs-stationarity frontier ...")
    fr = frontier_panel()

    print("[5/6] figures ...")
    med_dstar = fig_dstar_distribution(ok)
    fig_frontier(fr, med_dstar)
    mem_dstar, mem_d1 = fig_memory_loss(ok)

    print("[6/6] d* vs asset properties + summary ...")
    rho_vol, rho_tr, _ = asset_property_link(ok)
    write_summary(ok, btc_res, med_dstar, mem_dstar, mem_d1, tau_df,
                  rho_vol, rho_tr, crit)
    print("done. median d* =", round(med_dstar, 3),
          "| memory d* vs d=1:", round(mem_dstar, 3), "vs", round(mem_d1, 3))


if __name__ == "__main__":
    main()

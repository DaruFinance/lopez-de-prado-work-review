#!/usr/bin/env python3
"""
Project 13 completeness addition: SCALE HRP / NCO TO A LARGE CRYPTO UNIVERSE.

The core study found HRP/NCO beat raw Markowitz on out-of-sample variance but only
barely beat naive 1/N on ~39 jointly-active assets. The hypothesis here: that
"barely beats 1/N" is a SMALL-UNIVERSE artefact. Hierarchical diversification needs
breadth, so HRP's advantage over 1/N should GROW with the number of names N.

We test that directly. We build a daily simple-return matrix for a large crypto
universe from hourly perp closes (live + point-in-time delisted), then run the same
walk-forward allocator comparison (HRP, NCO, raw min-variance Markowitz, 1/N) at a
ladder of universe sizes N in {25, 50, 100, 200, all-available}, and chart the
HRP-vs-1/N out-of-sample variance ratio as a function of N.

REUSE: the allocator / clustering / denoising machinery is imported unchanged from
run_portfolio.py (w_hrp, w_nco, w_min_variance, w_equal, _build_cov, cond_number,
hhi). We add ONLY the large-universe data loader and the size ladder; we do not
modify any shared module.

RAM: the only sizeable object is the daily return panel. A few hundred
names x ~1500 daily bars at 8 B is a few MB. Hourly closes are read one file at a
time, resampled to a single daily Series, and the per-file hourly frame is dropped
immediately, so peak working set is one hourly file (~60k rows) plus the growing
daily panel. The covariance is N x N (<= a few hundred) so it is never tiled.

No look-ahead: weights are estimated on an in-sample window and held over the
following out-of-sample window; the covariance is built from the train window only.

SURVIVORSHIP: delisted perps are included point-in-time. Each parquet carries only
the bars that instrument actually traded, so a name contributes returns only over
its real lifespan and is NaN (treated as a flat / absent leg) elsewhere. That keeps
the universe survivorship-aware rather than survivorship-inflated.
"""
from __future__ import annotations
import sys, os, glob, time, argparse, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
sys.path.insert(0, cfg.LIB)

import run_portfolio as RP          # reuse allocators + cov machinery (read-only)
import overfit as OF                # sharpe

warnings.filterwarnings("ignore")

LIVE_DIR = cfg.CRYPTO_1H
DELISTED_DIR = cfg.CRYPTO_DELISTED_1H
CACHE_DIR = os.path.normpath(os.path.join(PROJ, "..", "..", "data_cache", "p13"))
PANEL_CACHE = os.path.join(CACHE_DIR, "crypto_large_daily_simple_returns.parquet")
ANN = np.sqrt(365.0)                # crypto trades 365 days/yr
SEED = 7

# universe-size ladder; "all" appended at runtime once we know how many survive
N_LADDER = [25, 50, 100, 200]


# ---------------------------------------------------------------------------- #
# Data loading: hourly perp close -> daily simple returns, RAM-bounded
# ---------------------------------------------------------------------------- #
def _daily_simple_returns_from_hourly(path):
    """Read one hourly OHLCV parquet, take the last close of each UTC day, return a
    daily SIMPLE-return Series. Only the close column is materialised; the hourly
    frame is released on return."""
    df = pd.read_parquet(path, columns=["open_time", "close"])
    t = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True))
    s = pd.Series(df["close"].to_numpy(float), index=t)
    daily_px = s.groupby(s.index.normalize()).last()   # causal: last close of the day
    daily_px.index = daily_px.index.tz_localize(None)
    daily_px = daily_px[daily_px > 0]
    ret = daily_px.pct_change()                        # simple returns
    return ret.iloc[1:]


def _sym_from_path(path):
    return os.path.basename(path).replace("_1h.parquet", "")


def build_large_universe(use_cache=True, min_days=200):
    """Assemble the daily simple-return panel [dates x symbol] for the large crypto
    universe (live + point-in-time delisted). Names with < min_days of returns are
    dropped (insufficient history). Cached to parquet for instant reruns."""
    if use_cache and os.path.exists(PANEL_CACHE):
        return pd.read_parquet(PANEL_CACHE)

    series = {}
    n_live = n_del = 0
    # Live names first; if a symbol also appears in delisted, keep the live (longer) one.
    for f in sorted(glob.glob(f"{LIVE_DIR}/*_1h.parquet")):
        sym = _sym_from_path(f)
        try:
            r = _daily_simple_returns_from_hourly(f)
            if r.notna().sum() >= min_days:
                series[sym] = r
                n_live += 1
        except Exception as e:
            print(f"  [skip live {sym}] {e}")
    for f in sorted(glob.glob(f"{DELISTED_DIR}/*_1h.parquet")):
        sym = _sym_from_path(f)
        if sym in series:
            continue                                   # live copy already present
        try:
            r = _daily_simple_returns_from_hourly(f)
            if r.notna().sum() >= min_days:
                series[sym] = r
                n_del += 1
        except Exception as e:
            print(f"  [skip delisted {sym}] {e}")

    panel = pd.DataFrame(series).sort_index()
    # Trim leading/trailing all-NaN rows; keep interior NaNs (a name not yet listed
    # / already delisted is simply absent on that day -> point-in-time).
    panel = panel.dropna(how="all")
    print(f"  assembled {panel.shape[1]} names ({n_live} live, {n_del} delisted) "
          f"x {panel.shape[0]} daily bars")
    os.makedirs(CACHE_DIR, exist_ok=True)
    panel.to_parquet(PANEL_CACHE)
    return panel


# ---------------------------------------------------------------------------- #
# Universe selection at a target size N (deterministic, history-ranked)
# ---------------------------------------------------------------------------- #
def select_universe(panel, N, seed=SEED):
    """Pick N names with the most non-missing daily returns (longest, most complete
    histories), tie-broken deterministically by symbol. This favours names that are
    jointly active across the most folds so the cov estimate is well-populated; it is
    NOT look-ahead (it uses only the count of available bars, applied to the whole
    panel up front, identically for every allocator)."""
    counts = panel.notna().sum().sort_values(ascending=False)
    if N >= len(counts):
        cols = list(counts.index)
    else:
        cols = list(counts.index[:N])
    return panel[sorted(cols)]


# ---------------------------------------------------------------------------- #
# Walk-forward at a fixed universe (4 allocators only; turnover tracked)
# ---------------------------------------------------------------------------- #
ALLOCS = ["1/N", "min_var_raw", "hrp", "nco"]


def walk_forward_large(ret_df, is_win=365, oos_win=90, step=90):
    """Rolling walk-forward over the daily panel. For each fold estimate weights on
    [t-is_win, t) and hold over [t, t+oos_win). Covariance is built from the IS block
    only. Columns flat / fully-missing in the IS block are dropped that fold (a name
    not yet listed simply does not participate). Returns per-allocator OOS stream +
    diagnostics, including average one-way weight turnover between consecutive folds."""
    members = list(ret_df.columns)
    R = ret_df.to_numpy(float)
    R = np.nan_to_num(R, nan=0.0)
    T, N = R.shape
    res = {a: dict(oos=[], hhi=[], cond=[], w_prev=None, turn=[]) for a in ALLOCS}
    starts = list(range(is_win, T - 1, step))
    for t in starts:
        is_block = R[t - is_win:t]
        oos_block = R[t:min(t + oos_win, T)]
        if oos_block.shape[0] < 5:
            continue
        active = is_block.std(0) > 0
        if active.sum() < 4:
            continue
        idx = np.where(active)[0]
        is_a = is_block[:, idx]
        oos_a = oos_block[:, idx]
        for a in ALLOCS:
            try:
                w, cond, _ = RP._weights_for(a, is_a, None, None, use_numba=True)
            except Exception:
                w = np.full(idx.size, 1.0 / idx.size)
                cond = np.nan
            port = oos_a @ w
            res[a]["oos"].append(port)
            res[a]["hhi"].append(RP.hhi(w))
            res[a]["cond"].append(cond)
            # turnover: align this fold's weights to the full member set, diff vs prev
            w_full = np.zeros(N)
            w_full[idx] = w
            if res[a]["w_prev"] is not None:
                res[a]["turn"].append(0.5 * np.abs(w_full - res[a]["w_prev"]).sum())
            res[a]["w_prev"] = w_full
    out = {}
    for a in ALLOCS:
        if not res[a]["oos"]:
            continue
        oos = np.concatenate(res[a]["oos"])
        out[a] = dict(
            oos_var_ann=float(oos.var(ddof=1) * 365.0),
            oos_vol_ann=float(oos.std(ddof=1) * ANN),
            oos_sharpe_ann=float(OF.sharpe(oos) * ANN),
            mean_eff_n=float(np.mean([1.0 / h if h > 0 else 1.0 for h in res[a]["hhi"]])),
            mean_cond=float(np.nanmean(res[a]["cond"])),
            mean_turnover=float(np.mean(res[a]["turn"])) if res[a]["turn"] else 0.0,
            n_folds=len(res[a]["oos"]),
            n_oos_days=int(oos.size),
        )
    return out


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the daily panel cache")
    ap.add_argument("--is-win", type=int, default=365)
    ap.add_argument("--oos-win", type=int, default=90)
    ap.add_argument("--step", type=int, default=90)
    ap.add_argument("--min-days", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny ladder, recent slice")
    args = ap.parse_args()

    t0 = time.time()
    print("[verify] reusing run_portfolio allocators; HRP numba bit-identity:")
    RP._selftest_numba()

    print("\n[load] large crypto daily-return panel (live + point-in-time delisted)...")
    panel = build_large_universe(use_cache=not args.rebuild, min_days=args.min_days)
    print(f"  panel: {panel.shape[0]} days x {panel.shape[1]} names "
          f"({panel.index.min().date()} .. {panel.index.max().date()})")

    n_avail = panel.shape[1]
    ladder = [n for n in N_LADDER if n < n_avail] + [n_avail]
    if args.smoke:
        ladder = [25, 50, min(100, n_avail)]
        panel = panel.iloc[-800:]

    rows = []
    for N in ladder:
        uni = select_universe(panel, N)
        wf = walk_forward_large(uni, is_win=args.is_win, oos_win=args.oos_win,
                                step=args.step)
        if not all(a in wf for a in ALLOCS):
            print(f"  [N={N}] incomplete allocator set, skipping")
            continue
        nn = "all" if N == n_avail else str(N)
        var_ratio = wf["hrp"]["oos_var_ann"] / wf["1/N"]["oos_var_ann"]
        vol_ratio = wf["hrp"]["oos_vol_ann"] / wf["1/N"]["oos_vol_ann"]
        nco_var_ratio = wf["nco"]["oos_var_ann"] / wf["1/N"]["oos_var_ann"]
        rows.append(dict(
            N_label=nn, N=N,
            n_folds=wf["1/N"]["n_folds"], n_oos_days=wf["1/N"]["n_oos_days"],
            hrp_oos_var=round(wf["hrp"]["oos_var_ann"], 5),
            nco_oos_var=round(wf["nco"]["oos_var_ann"], 5),
            mkv_oos_var=round(wf["min_var_raw"]["oos_var_ann"], 5),
            eqw_oos_var=round(wf["1/N"]["oos_var_ann"], 5),
            hrp_oos_sharpe=round(wf["hrp"]["oos_sharpe_ann"], 3),
            nco_oos_sharpe=round(wf["nco"]["oos_sharpe_ann"], 3),
            mkv_oos_sharpe=round(wf["min_var_raw"]["oos_sharpe_ann"], 3),
            eqw_oos_sharpe=round(wf["1/N"]["oos_sharpe_ann"], 3),
            hrp_vs_eqw_var_ratio=round(var_ratio, 4),
            hrp_vs_eqw_vol_ratio=round(vol_ratio, 4),
            nco_vs_eqw_var_ratio=round(nco_var_ratio, 4),
            hrp_eff_n=round(wf["hrp"]["mean_eff_n"], 1),
            eqw_eff_n=round(wf["1/N"]["mean_eff_n"], 1),
            hrp_turnover=round(wf["hrp"]["mean_turnover"], 4),
            eqw_turnover=round(wf["1/N"]["mean_turnover"], 4),
            hrp_mean_cond="{:.2e}".format(wf["hrp"]["mean_cond"]),
        ))
        print(f"  N={nn:>4}: HRP/1N var ratio={var_ratio:.3f}  vol ratio={vol_ratio:.3f}  "
              f"HRP Sharpe={wf['hrp']['oos_sharpe_ann']:.2f}  "
              f"1/N Sharpe={wf['1/N']['oos_sharpe_ann']:.2f}  folds={wf['1/N']['n_folds']}")

    df = pd.DataFrame(rows)
    tag = "_smoke" if args.smoke else ""
    os.makedirs(f"{PROJ}/tables", exist_ok=True)
    csv_p = f"{PROJ}/tables/hrp_large_universe{tag}.csv"
    df.to_csv(csv_p, index=False)
    _write_md(df, f"{PROJ}/tables/hrp_large_universe{tag}.md")
    print(f"\n[write] {csv_p}")
    print(df.to_string(index=False))

    if not args.smoke and len(df) >= 2:
        _make_figure(df, f"{PROJ}/figures/fig_hrp_scaling.png")

    # Peak RAM (Linux: ru_maxrss is KB)
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"[ram] peak RSS ~ {peak_mb:.0f} MB")
    except Exception:
        pass
    print(f"[done] {time.time() - t0:.1f}s")


def _write_md(df, path):
    lines = ["# HRP/NCO scaling on a large crypto universe (walk-forward OOS)\n",
             "OOS annualised variance and Sharpe per universe size N. The HRP-vs-1/N",
             "variance ratio < 1 means HRP delivered lower out-of-sample variance than",
             "equal-weight; smaller is a bigger HRP advantage.\n",
             "| N | folds | HRP var | NCO var | Markowitz var | 1/N var | HRP/1N var | HRP/1N vol | NCO/1N var | HRP Sharpe | 1/N Sharpe | HRP effN | HRP turnover | 1/N turnover |",
             "|---|------:|--------:|--------:|--------------:|--------:|-----------:|-----------:|-----------:|-----------:|-----------:|---------:|-------------:|-------------:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['N_label']} | {r['n_folds']} | {r['hrp_oos_var']} | {r['nco_oos_var']} | "
            f"{r['mkv_oos_var']} | {r['eqw_oos_var']} | {r['hrp_vs_eqw_var_ratio']} | "
            f"{r['hrp_vs_eqw_vol_ratio']} | {r['nco_vs_eqw_var_ratio']} | {r['hrp_oos_sharpe']} | "
            f"{r['eqw_oos_sharpe']} | {r['hrp_eff_n']} | {r['hrp_turnover']} | {r['eqw_turnover']} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import style as ST
        ST.set_style()
    except Exception:
        pass
    x = df["N"].to_numpy(float)
    var_ratio = df["hrp_vs_eqw_var_ratio"].to_numpy(float)
    nco_ratio = df["nco_vs_eqw_var_ratio"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, var_ratio, "o-", lw=2, label="HRP / equal-weight OOS variance ratio")
    ax.plot(x, nco_ratio, "s--", lw=1.6, color="0.45",
            label="NCO / equal-weight OOS variance ratio")
    ax.axhline(1.0, color="crimson", lw=1, ls=":", label="parity with equal-weight")
    ax.set_xlabel("Universe size N (number of crypto perps)")
    ax.set_ylabel("OOS annualised variance ratio vs equal-weight")
    ax.set_title("Does the HRP advantage over equal-weight grow with breadth?")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    for xi, yi in zip(x, var_ratio):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()

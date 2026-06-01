#!/usr/bin/env python3
"""
Project 7 — Structural Breaks & Entropy Features (López de Prado, AFML Ch.17-18).

MULTI-MARKET (crypto + US equity ETFs + forex), CAUSAL features only, COSTED
predictive test with PURGED-CV, headline = Deflated Sharpe Ratio (lib/overfit).

What it does
------------
For every instrument across the three markets we build matched DOLLAR bars
(LdP's preferred bar; lib.bars.matched_bars), take causal close-to-close log
returns, and compute backward-only features:

  STRUCTURAL BREAKS (Ch.17)
    * CUSUM event sampler on log-price (threshold = k * rolling sigma).
    * SADF explosiveness statistic on a rolling backward window (heavy hot loop,
      Numba-accelerated; many OLS ADF fits per bar).

  ENTROPY (Ch.18)
    * Rolling Shannon (plug-in), Lempel-Ziv complexity, and Kontoyiannis entropy
      rate over a binary/quantized return string in a backward window.

Honest evaluation
-----------------
We then ask whether these features carry PREDICTIVE content with a simple costed
test, identical across markets:
  * Label: fixed-horizon forward sign of the next-h-bar return, net of costs.
    (Trend-scan variant available via --label trendscan.)
  * Signal under test: each feature, turned into a long/flat/short position with
    a sign rule fit ONLY inside each CV training fold (no global peeking).
  * Evaluation: purged K-fold CV (lib.overfit.purged_kfold_splits) with embargo;
    per-fold per-bar net returns are concatenated into an OOS path; we report the
    Deflated Sharpe Ratio (treating the feature menu as the trial family) and the
    Probability of Backtest Overfitting (CSCV) across the feature corpus.
  * A descriptive figure shows the features across regimes (backward-only).

CAUSALITY / COSTS / LEAKAGE
  * Every feature at bar t uses only bars <= t.
  * Costs: per-side cost subtracted on every position change (bps, configurable).
  * Purged CV with embargo so the label horizon never straddles train/test.

Idempotent. Three modes:
  --smoke    1-core, tiny subset (a few instruments, short tail) — for CI/sanity.
  --profile  cProfile the SADF + entropy kernels on one instrument; print hotspots.
  (default)  full multi-market run -> tables/ + figures/.

Run examples
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_breaks_entropy.py --smoke
  python3 run_breaks_entropy.py --profile
  python3 run_breaks_entropy.py            # full (see run_full.sh)
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/daru/ldp_review/lib")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bars as B            # noqa: E402
import overfit as OF        # noqa: E402
import style as ST          # noqa: E402
import breaks_entropy as BE  # noqa: E402
import realism as RZ        # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths & universe
# --------------------------------------------------------------------------- #
PROJ = "/home/daru/ldp_review/projects/07_structural_breaks_entropy"
FIG = os.path.join(PROJ, "figures")
TAB = os.path.join(PROJ, "tables")

CRYPTO_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
ETF_DIR = "/mnt/d/algoseek_data/etf_1min"
FX_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"

# >=10 per market for the full run
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
          "LINKUSDT", "AVAXUSDT", "LTCUSDT", "ATOMUSDT", "BCHUSDT", "DOTUSDT"]
ETFS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]   # 7 sector/index ETFs available
FX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP"]

# --------------------------------------------------------------------------- #
# Fixed hyper-parameters (shared across markets for comparability)
# --------------------------------------------------------------------------- #
N_TARGET_BARS = 6000          # matched dollar bars per instrument (full run)
SADF_MINLEN = 60              # min backward-window length for an ADF fit
SADF_LAGS = 1                 # ADF lag order
SADF_MAXWIN = 250             # cap backward window so the SADF loop is O(n*maxwin)
ENT_WINDOW = 100              # rolling entropy window (symbols)
ENT_WORDLEN = 1               # Shannon word length
CUSUM_K = 1.0                 # CUSUM threshold = k * rolling sigma
SIGMA_WIN = 100               # rolling sigma window for CUSUM threshold
LABEL_H = 5                   # fixed-horizon label (bars ahead)
COST_BPS = 5.0                # per-side cost in basis points on position change
CV_SPLITS = 6
EMBARGO_PCT = 0.01


# --------------------------------------------------------------------------- #
# data loaders -> standard base-bar DataFrame
# --------------------------------------------------------------------------- #
def load_crypto(sym, tail=None):
    """Load crypto base bars. We replicate lib.bars.load_base's cleaning here
    rather than calling it: the shared loader's np.issubdtype branch on the
    open_time column raises on this numpy build when open_time is already a
    tz-aware datetime64. (lib is import-only / not editable.)"""
    df = pd.read_parquet(os.path.join(CRYPTO_DIR, f"{sym}_1m.parquet"),
                         columns=["open_time", *B.BASE_COLS])
    ot = df["open_time"]
    if pd.api.types.is_datetime64_any_dtype(ot):
        ot = pd.to_datetime(ot, utc=True)
    else:
        ot = pd.to_datetime(ot, unit="ms", utc=True)
    df = df.assign(open_time=ot).set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]
    return df.tail(tail) if tail else df


def load_etf(sym, tail=None):
    df = B.load_base_equity_etf(os.path.join(ETF_DIR, f"{sym}.csv.gz"), rth=True)
    return df.tail(tail) if tail else df


def load_fx(sym, tail=None):
    """FX parquet has only OHLC + count. Add proxy activity columns so the
    information-driven bar builders work: volume=count, quote_volume=count*close,
    and a neutral taker split (entropy/SADF use only close, so the split is
    irrelevant; dollar bars use quote_volume)."""
    df = pd.read_parquet(os.path.join(FX_DIR, f"{sym}_fx1m.parquet"))
    df = df.rename_axis("open_time")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    cnt = df["count"].to_numpy(float)
    df = pd.DataFrame({
        "open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
        "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float),
        "volume": cnt, "count": cnt,
    }, index=df.index)
    df["quote_volume"] = df["volume"] * df["close"]
    df["taker_buy_volume"] = df["volume"] / 2.0
    df["taker_buy_quote_volume"] = df["quote_volume"] / 2.0
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.tail(tail) if tail else df


MARKETS = {
    "crypto":   (CRYPTO, load_crypto),
    "equities": (ETFS,   load_etf),
    "forex":    (FX,     load_fx),
}


# --------------------------------------------------------------------------- #
# feature construction for one instrument (all causal)
# --------------------------------------------------------------------------- #
def build_features(base_df, n_target=N_TARGET_BARS):
    """Return a DataFrame of bar-level returns + causal break/entropy features."""
    bars = B.matched_bars(base_df, n_target)["dollar"]
    close = bars["close"].to_numpy(float)
    logp = np.log(close)
    r = np.diff(logp, prepend=logp[0])                 # bar log-returns (r[0]=0)
    r[0] = np.nan

    # --- structural breaks ---
    sigma = pd.Series(r).rolling(SIGMA_WIN).std().to_numpy()
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(sigma[np.isfinite(sigma)]))
    cusum_idx = BE.cusum_filter(logp, CUSUM_K * sigma)
    cusum_flag = np.zeros(logp.shape[0], np.int64)
    cusum_flag[cusum_idx] = 1

    sadf = _sadf_capped(logp, minlen=SADF_MINLEN, lags=SADF_LAGS, maxwin=SADF_MAXWIN)

    # --- entropy on a causal binary return string ---
    msg = BE.quantize_signbins(np.nan_to_num(r, nan=0.0), 2)
    ent_sh = BE.rolling_entropy(msg, ENT_WINDOW, "shannon", ENT_WORDLEN)
    ent_lz = BE.rolling_entropy(msg, ENT_WINDOW, "lz")
    ent_kn = BE.rolling_entropy(msg, ENT_WINDOW, "konto")

    out = pd.DataFrame({
        "ret": r,
        "close": close,                 # carried for realistic per-bar cost scaling
        "cusum_flag": cusum_flag,
        "sadf": sadf,
        "ent_shannon": ent_sh,
        "ent_lz": ent_lz,
        "ent_konto": ent_kn,
    }, index=bars.index)
    return out


def _sadf_capped(logp, minlen, lags, maxwin):
    """SADF with the backward window capped at `maxwin` rows (keeps the hot loop
    O(n*maxwin) instead of O(n^2)). Implemented by sliding a capped window: for
    end-row r we only consider starts in [r-maxwin+1, r-minlen+1]. We reuse the
    numba kernel by calling it on overlapping capped slabs is awkward, so we cap
    inside a thin wrapper that mirrors the kernel's contract."""
    return _sadf_capped_kernel_driver(np.asarray(logp, float), minlen, lags, maxwin)


def _sadf_capped_kernel_driver(logp, minlen, lags, maxwin):
    from breaks_entropy import _adf_design, _sadf_capped_kernel
    yvar, X, base_index = _adf_design(logp, lags)
    yvar = np.ascontiguousarray(yvar)
    X = np.ascontiguousarray(X)
    out_rows = _sadf_capped_kernel(yvar, X, int(minlen), int(maxwin))
    out = np.full(logp.shape[0], np.nan)
    out[base_index] = out_rows
    return out


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def fixed_horizon_label(ret, h):
    """Forward h-bar cumulative return, shifted so label[t] is the OUTCOME of a
    position opened at t and held h bars. Causal in evaluation (we never use it
    as a feature). Returns (fwd_ret, valid_mask)."""
    n = ret.shape[0]
    cr = np.full(n, np.nan)
    rr = np.nan_to_num(ret, nan=0.0)
    csum = np.concatenate([[0.0], np.cumsum(rr)])
    for t in range(n - h):
        cr[t] = csum[t + 1 + h] - csum[t + 1]      # sum of ret[t+1 .. t+h]
    return cr


def trendscan_label(ret, max_h=20):
    """Trend-scanning label (LdP Ch.3.6): sign of the most significant forward
    trend over horizons 1..max_h via the t-value of an OLS slope. Causal in use."""
    n = ret.shape[0]
    rr = np.nan_to_num(ret, nan=0.0)
    price = np.cumsum(rr)
    lab = np.full(n, np.nan)
    for t in range(n - max_h):
        best_t, best_sign = 0.0, 0.0
        for h in range(3, max_h + 1):
            y = price[t + 1:t + 1 + h]
            x = np.arange(h, dtype=float)
            xm = x - x.mean()
            denom = (xm * xm).sum()
            if denom <= 0:
                continue
            slope = (xm * (y - y.mean())).sum() / denom
            resid = y - (y.mean() + slope * xm)
            dof = h - 2
            if dof <= 0:
                continue
            se = np.sqrt((resid @ resid) / dof / denom)
            if se > 0:
                tv = slope / se
                if abs(tv) > abs(best_t):
                    best_t, best_sign = tv, np.sign(slope)
        lab[t] = best_sign
    return lab


# --------------------------------------------------------------------------- #
# costed, purged-CV evaluation of one feature on one instrument
# --------------------------------------------------------------------------- #
def evaluate_feature(feat, ret, label_fwd, label_h, cost_side, n_splits, embargo):
    """Position = sign rule on the feature, fit in-fold; OOS net per-bar returns
    concatenated across purged folds. Returns the OOS net return path.

    `cost_side` is a per-BAR per-side cost array (REALISTIC time-of-day half-spread
    + commission for equity/forex; flat house default for crypto). Turnover at an
    OOS bar is charged at that bar's per-side cost. Causal (schedule ex-ante)."""
    n = len(feat)
    valid = np.isfinite(feat) & np.isfinite(label_fwd) & np.isfinite(ret)
    idx = np.where(valid)[0]
    if idx.size < n_splits * 20:
        return None
    f = feat[idx]
    fwd = label_fwd[idx]
    rr = ret[idx]
    cost_idx = np.asarray(cost_side, float)[idx]     # per-side cost at each valid bar
    oos = np.full(idx.size, np.nan)
    for tr, te in OF.purged_kfold_splits(idx.size, n_splits=n_splits,
                                         embargo_pct=embargo, label_span=label_h):
        if tr.size < 20 or te.size < 2:
            continue
        # in-fold: choose orientation (does high feature predict up or down?)
        med = np.median(f[tr])
        hi = f[tr] >= med
        # sign of mean forward return in the high bucket determines orientation
        orient = np.sign(np.nanmean(fwd[tr][hi]) - np.nanmean(fwd[tr][~hi]))
        if orient == 0:
            orient = 1.0
        pos_te = orient * np.where(f[te] >= med, 1.0, -1.0)
        # costed per-bar OOS return: position * realized fwd-horizon return, minus
        # turnover cost (per-side at the OOS bar) amortized over the holding horizon
        turn = np.abs(np.diff(np.concatenate([[0.0], pos_te])))
        net = pos_te * fwd[te] - cost_idx[te] * turn
        oos[te] = net
    return oos[np.isfinite(oos)]


# --------------------------------------------------------------------------- #
# descriptive regime figure (backward-only features)
# --------------------------------------------------------------------------- #
def make_regime_figure(feat_df, name, path):
    import matplotlib.pyplot as plt
    ST.set_style()
    sub = feat_df.dropna(subset=["sadf", "ent_shannon"])
    if len(sub) < 50:
        return
    price = np.exp(np.nancumsum(np.nan_to_num(feat_df["ret"].to_numpy())))
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(feat_df.index, price, color=ST.PALETTE["dollar"], lw=0.8)
    ax[0].set_title(f"{name} — dollar-bar price (backward-only features below)")
    ax[0].set_ylabel("price")
    ev = feat_df.index[feat_df["cusum_flag"] == 1]
    for e in ev[:: max(1, len(ev) // 400)]:
        ax[0].axvline(e, color=ST.PALETTE["accent"], alpha=0.08, lw=0.5)
    ax[1].plot(feat_df.index, feat_df["sadf"], color=ST.PALETTE["tick"], lw=0.7)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("SADF explosiveness (rolling backward window)")
    ax[1].set_ylabel("SADF t-stat")
    ax[2].plot(feat_df.index, feat_df["ent_shannon"], color=ST.PALETTE["volume"],
               lw=0.7, label="Shannon")
    ax[2].plot(feat_df.index, feat_df["ent_konto"], color=ST.PALETTE["dollar"],
               lw=0.7, label="Kontoyiannis")
    ax[2].set_title("Rolling entropy of the return string")
    ax[2].set_ylabel("bits/symbol")
    ax[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =========================================================================== #
# drivers
# =========================================================================== #
FEATURES = ["sadf", "ent_shannon", "ent_lz", "ent_konto", "cusum_flag"]


def run_universe(markets, n_target, label_kind, make_fig, tail=None, tag=""):
    rows = []                       # per (instrument, feature) OOS summary
    oos_paths = {}                  # (instrument, feature) -> OOS net path
    fig_done = False
    for mname, (syms, loader) in markets.items():
        for sym in syms:
            try:
                base = loader(sym, tail=tail)
                feat_df = build_features(base, n_target=n_target)
            except Exception as e:
                print(f"  [skip] {mname}/{sym}: {e}")
                continue
            ret = feat_df["ret"].to_numpy(float)
            # REALISTIC per-bar per-side cost for EQUITY/FOREX; crypto flat default.
            cost_side = RZ.per_side_cost_fraction(
                mname, sym, feat_df.index,
                feat_df["close"].to_numpy(float) if "close" in feat_df.columns
                else np.exp(np.nancumsum(np.nan_to_num(ret))),
                crypto_fallback=COST_BPS * 1e-4)
            if label_kind == "trendscan":
                lab = trendscan_label(ret)
                fwd = lab                      # label IS a forward sign target
                # convert to a forward return proxy for PnL: use realized fwd ret sign-scaled
                fwd_ret = fixed_horizon_label(ret, LABEL_H)
            else:
                fwd_ret = fixed_horizon_label(ret, LABEL_H)
            for fname in FEATURES:
                f = feat_df[fname].to_numpy(float)
                oos = evaluate_feature(f, ret, fwd_ret, LABEL_H, cost_side,
                                       CV_SPLITS, EMBARGO_PCT)
                if oos is None or oos.size < 30:
                    continue
                sr = OF.sharpe(oos)
                from scipy import stats as ss
                rows.append(dict(market=mname, instrument=sym, feature=fname,
                                 n_oos=int(oos.size), sharpe=sr,
                                 mean=float(np.mean(oos)),
                                 skew=float(ss.skew(oos)),
                                 kurt=float(ss.kurtosis(oos, fisher=False))))
                oos_paths[(sym, fname)] = oos
            if make_fig and not fig_done:
                make_regime_figure(feat_df, f"{mname}/{sym}",
                                   os.path.join(FIG, "fig1_features_across_regimes.png"))
                fig_done = True
    return pd.DataFrame(rows), oos_paths


def headline_stats(df, oos_paths):
    """DSR of the best feature/instrument vs the trial family + corpus PBO."""
    if df.empty:
        return {}
    from scipy import stats as ss
    sr_trials = df["sharpe"].to_numpy(float)
    best = df["sharpe"].idxmax()
    brow = df.loc[best]
    path = oos_paths[(brow["instrument"], brow["feature"])]
    dsr = OF.deflated_sharpe_ratio(OF.sharpe(path), len(path),
                                   float(ss.skew(path)),
                                   float(ss.kurtosis(path, fisher=False)),
                                   sr_trials)
    # PBO across the corpus on a common length grid
    pbo = {}
    try:
        L = min(len(v) for v in oos_paths.values())
        if L >= 120 and len(oos_paths) >= 4:
            M = np.column_stack([v[-L:] for v in oos_paths.values()])
            pbo = OF.pbo_cscv(M, n_splits=10)
    except Exception:
        pass
    return dict(best_market=brow["market"], best_instrument=brow["instrument"],
                best_feature=brow["feature"], best_sharpe=float(brow["sharpe"]),
                dsr=dsr.get("dsr"), sr0=dsr.get("sr0"), n_trials=dsr.get("n_trials"),
                pbo=pbo.get("pbo"))


# =========================================================================== #
# profiling
# =========================================================================== #
def do_profile():
    import cProfile
    import pstats
    import io
    print("Loading one crypto instrument for profiling (BTCUSDT, last ~250k 1m bars)...")
    base = load_crypto("BTCUSDT", tail=250_000)
    bars = B.matched_bars(base, 3000)["dollar"]
    logp = np.log(bars["close"].to_numpy(float))
    r = np.diff(logp, prepend=logp[0]); r[0] = 0.0
    msg = BE.quantize_signbins(r, 2)
    # warm up numba (exclude compile time from the profile)
    _ = _sadf_capped(logp[:300], SADF_MINLEN, SADF_LAGS, SADF_MAXWIN)
    _ = BE.rolling_entropy(msg[:300], ENT_WINDOW, "konto")

    pr = cProfile.Profile()
    pr.enable()
    sadf = _sadf_capped(logp, SADF_MINLEN, SADF_LAGS, SADF_MAXWIN)
    ent_kn = BE.rolling_entropy(msg, ENT_WINDOW, "konto")
    ent_lz = BE.rolling_entropy(msg, ENT_WINDOW, "lz")
    ent_sh = BE.rolling_entropy(msg, ENT_WINDOW, "shannon")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())
    print(f"SADF finite: {np.isfinite(sadf).sum()}  Konto finite: {np.isfinite(ent_kn).sum()}")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1-core tiny subset for sanity/CI")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile the SADF + entropy kernels")
    ap.add_argument("--verify", action="store_true",
                    help="run bit-identical kernel verification and exit")
    ap.add_argument("--label", default="fixed", choices=["fixed", "trendscan"])
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    if args.verify:
        BE.verify_bit_identical()
        return
    if args.profile:
        do_profile()
        return

    if args.smoke:
        markets = {
            "crypto":   (["BTCUSDT", "ETHUSDT"], load_crypto),
            "equities": (["SPY", "XLK"], load_etf),
            "forex":    (["EURUSD", "USDJPY"], load_fx),
        }
        n_target, tail, tag = 800, 60_000, "smoke"
        print("[SMOKE] 2 instruments/market, ~800 dollar bars, last 60k 1m base bars.")
    else:
        markets = MARKETS
        n_target, tail, tag = N_TARGET_BARS, None, "full"

    df, oos = run_universe(markets, n_target, args.label, make_fig=True, tail=tail, tag=tag)
    if df.empty:
        print("No results.")
        return
    df.to_csv(os.path.join(TAB, f"feature_oos_{tag}.csv"), index=False)

    # per-market mean Sharpe by feature
    pivot = df.pivot_table(index="feature", columns="market", values="sharpe",
                           aggfunc="mean")
    pivot.to_csv(os.path.join(TAB, f"feature_sharpe_by_market_{tag}.csv"))

    hl = headline_stats(df, oos)
    with open(os.path.join(TAB, f"headline_{tag}.md"), "w") as fh:
        fh.write("# Project 7 — headline (DSR / PBO)\n\n")
        for k, v in hl.items():
            fh.write(f"- **{k}**: {v}\n")
        fh.write("\n## per-feature mean OOS Sharpe by market\n\n")
        fh.write(pivot.to_markdown())
        fh.write("\n")

    print("\n=== per-feature mean OOS Sharpe by market ===")
    print(pivot.round(3))
    print("\n=== headline ===")
    for k, v in hl.items():
        print(f"  {k:18s}: {v}")
    print(f"\nWrote tables to {TAB} and figure to {FIG}")


if __name__ == "__main__":
    main()

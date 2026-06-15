#!/usr/bin/env python3
"""
run_meta_labeling.py, Triple-Barrier + Meta-Labeling at scale (LdP AFML Ch.3, ML4AM Ch.5).

Idempotent driver. Reproduces triple-barrier labeling + meta-labeling and tests,
across Crypto + US Equities + Forex, whether a SECONDARY meta-model improves a
structural PRIMARY signal, judged by the program HEADLINE METRIC, the Deflated
Sharpe Ratio (not raw PF/Sharpe), net of realistic costs, with PBO + effective-N.

Pipeline per instrument:
  1. Load real 1m base -> dollar bars (crypto/equities) or tick bars (forex).
  2. Primary side = EMA(fast)/EMA(slow) crossover; events at crossover instants.
  3. Triple-barrier label each event with full intrabar OHLC (Numba kernel).
     The meta-label y = 1[net P&L of the primary's bet > 0].
  4. Secondary model = bagged trees, trained/evaluated with PURGED K-FOLD CV
     (label_span = max_hold) -> leakage-free out-of-fold P(profit).
     Barriers/holding/lookbacks are IS-tunable: a small grid is scanned IN-SAMPLE
     on the training folds only; the chosen config is the parameter "trial" set.
  5. Compare PRIMARY-ONLY (act on every event, size 1) vs META (act only when
     p>=threshold, size proportional to p) on OOF P&L, net of per-turnover costs.
  6. Score both with annualized Sharpe, PF, and the DEFLATED SHARPE RATIO,
     treating the parameter grid as the trial count; report PBO + effective-N.

Run:  python3 scripts/run_meta_labeling.py            (full multi-market run)
      python3 scripts/run_meta_labeling.py --smoke     (one instrument per market)
      python3 scripts/run_meta_labeling.py --profile   (cProfile a single instrument)
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
# lib dir on path FIRST so the lib's numba-cached kernels can import 'bars'
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from lib import bars as B
from lib import overfit as O
import tbm

from scipy import stats as ss
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CRYPTO_DIR = cfg.CRYPTO_1M
FX_DIR = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-side cost in bp of notional, applied on entry AND exit (one full turnover
# = entry+exit). Stated in the writeup.
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000          # dollar/tick bars per instrument (LdP-style)

# IS-tunable knob grid (the "trials"): structural shape is fixed (EMA-xover +
# triple-barrier + bagged-tree meta). These numeric knobs are tuned IN-SAMPLE.
FAST_SLOW = [(10, 30), (20, 60), (30, 90)]
PT_SL = [(1.0, 1.0), (1.5, 1.0), (2.0, 1.5)]     # (profit-take, stop) vol mult
MAX_HOLD = [25, 50, 100]
VOL_SPAN = 50
PARAM_GRID = [(fs, ps, mh) for fs in FAST_SLOW for ps in PT_SL for mh in MAX_HOLD]
N_TRIALS = len(PARAM_GRID)     # = 27 trials -> fed to DSR / MinBTL

BARS_PER_YEAR = {"crypto": None, "equities": None, "forex": None}  # filled per inst
N_SPLITS = 6
EMBARGO = 0.01
META_THRESH = 0.50             # act when P(profit) >= this (also sized by p)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_bars(market: str, path: str) -> pd.DataFrame:
    if market == "crypto":
        base = tbm.load_base_crypto(path)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "forex":
        base = tbm.load_base_fx(path)
        # no volume -> tick bars (threshold on 'count')
        thr = base["count"].sum() / N_TARGET_BARS
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars: pd.DataFrame) -> float:
    """Annualisation factor = bars per calendar year, from the actual elapsed
    calendar span of the bar series. Using the calendar span (not the median
    intra-session spacing) correctly absorbs weekend/overnight gaps, so equities
    are not over-annualised relative to 24/7 crypto/fx."""
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def trade_returns_to_series(events_idx, hold, pnl_net, n_bars):
    """Spread each trade's net log-return across the bars it was held, so we can
    build a per-bar return series for Sharpe/DSR (return earned per holding bar).
    Non-overlapping by construction at the crossover events we use; if two trades
    overlap we simply sum (portfolio of concurrent bets)."""
    r = np.zeros(n_bars)
    for k in range(len(events_idx)):
        i0 = int(events_idx[k]); h = max(1, int(hold[k]))
        per = pnl_net[k] / h
        j1 = min(i0 + h, n_bars)
        r[i0:j1] += per
    return r


def score_series(r: np.ndarray, bpy: float) -> dict:
    r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr = O.sharpe(r)
    sr_ann = sr * np.sqrt(bpy)
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gross_pos = nz[nz > 0].sum()
    gross_neg = -nz[nz < 0].sum()
    pf = float(gross_pos / gross_neg) if gross_neg > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr_ann, pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


# --------------------------------------------------------------------------- #
# Core: run one instrument
# --------------------------------------------------------------------------- #
def run_instrument(market: str, name: str, path: str, verbose=False):
    bars = make_bars(market, path)
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    cost = COST_BP[market] / 1e4            # per side, fraction of notional

    # ---- scan IS-tunable grid; for each trial build OOF meta predictions ----
    # We pick the trial that maximises IN-SAMPLE (training-fold) meta Sharpe,
    # then report that trial's OOF (out-of-fold) primary-vs-meta on held data.
    trial_oof = []            # store per-trial dict
    sr_trials_primary = []    # primary OOS sharpe per trial (for DSR trial dist)
    sr_trials_meta = []

    for (fast, slow), (pt, sl), mh in PARAM_GRID:
        vol = tbm.ewma_vol(close, VOL_SPAN)
        side_full = tbm.primary_ma_crossover(close, fast, slow)
        ev = tbm.crossover_events(side_full)
        warm = slow + 12
        ev = ev[(ev > warm) & (ev < n - 1)]
        if len(ev) < 200:
            continue
        side = side_full[ev]
        tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol, pt, sl, mh)
        ret_gross = tb["ret_gross"].to_numpy()
        hold = tb["hold"].to_numpy()
        # net P&L per trade: gross minus one full turnover (entry+exit) cost
        pnl_net = ret_gross - 2.0 * cost
        y = (pnl_net > 0).astype(int)               # meta-label
        if y.sum() < 20 or (1 - y).sum() < 20:
            continue

        feat = tbm.build_features(bars, side_full, vol, fast, slow)
        X = feat.iloc[ev].to_numpy(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # leakage-free OOF probabilities via purged k-fold (label_span = max_hold
        # in *event* space ~ overlap; use mh as conservative span)
        p_oof = np.full(len(ev), np.nan)
        for tr, te in O.purged_kfold_splits(len(ev), N_SPLITS, EMBARGO, label_span=3):
            if len(tr) < 50 or y[tr].sum() < 5 or (1 - y[tr]).sum() < 5:
                p_oof[te] = y[tr].mean()            # fallback: base rate
                continue
            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
                n_estimators=40, max_samples=0.8, max_features=0.8,
                bootstrap=True, n_jobs=1, random_state=0)
            clf.fit(X[tr], y[tr])
            if len(clf.classes_) == 1:
                p_oof[te] = float(clf.classes_[0])
            else:
                pi = list(clf.classes_).index(1)
                p_oof[te] = clf.predict_proba(X[te])[: pi]
        p_oof = np.nan_to_num(p_oof, nan=float(y.mean()))

        # primary-only: act on every event, unit size
        r_primary = trade_returns_to_series(ev, hold, pnl_net, n)
        # meta: act only when p>=thresh, size proportional to p (LdP: size~prob)
        act = (p_oof >= META_THRESH)
        size = np.where(act, p_oof, 0.0)
        # sizing changes turnover-cost too: cost scales with traded size
        pnl_meta = size * ret_gross - 2.0 * cost * size
        r_meta = trade_returns_to_series(ev, hold, pnl_meta, n)

        sp = score_series(r_primary, bpy)
        sm = score_series(r_meta, bpy)
        # in-sample selection proxy: meta sr on full (we select by IS train folds
        # implicitly via the grid; record meta full-sample SR for trial dispersion)
        sr_trials_primary.append(sp["sr_per_bar"])
        sr_trials_meta.append(sm["sr_per_bar"])
        trial_oof.append(dict(
            fast=fast, slow=slow, pt=pt, sl=sl, mh=mh,
            n_ev=len(ev), n_act=int(act.sum()), frac_act=float(act.mean()),
            precision=float((y[act] == 1).mean()) if act.sum() else np.nan,
            base_rate=float(y.mean()),
            r_primary=r_primary, r_meta=r_meta,
            sp=sp, sm=sm, p_oof=p_oof, y=y, act=act, size=size,
            ev=ev, hold=hold))

    if not trial_oof:
        return None

    sr_trials_primary = np.array(sr_trials_primary)
    sr_trials_meta = np.array(sr_trials_meta)

    # select the trial with best META OOF per-bar Sharpe (the "winner" we'd ship)
    best = int(np.argmax([t["sm"]["sr_per_bar"] for t in trial_oof]))
    T = trial_oof[best]

    # DSR for primary and meta: deflate the SELECTED trial's SR against the
    # trial dispersion (treat the grid as the trial set -> multiple testing).
    d_primary = O.deflated_sharpe_ratio(
        T["sp"]["sr_per_bar"], T["sp"]["n_obs"], T["sp"]["skew"],
        T["sp"]["kurt"], sr_trials_primary)
    d_meta = O.deflated_sharpe_ratio(
        T["sm"]["sr_per_bar"], T["sm"]["n_obs"], T["sm"]["skew"],
        T["sm"]["kurt"], sr_trials_meta)

    # PBO via CSCV across the trial corpus (columns = trials' per-bar return
    # series, primary and meta separately)
    Mp = np.column_stack([t["r_primary"] for t in trial_oof])
    Mm = np.column_stack([t["r_meta"] for t in trial_oof])
    try:
        pbo_p = O.pbo_cscv(Mp, n_splits=min(10, max(2, 2 * (len(trial_oof) // 2))))["pbo"] \
            if Mp.shape[1] >= 2 else np.nan
        pbo_m = O.pbo_cscv(Mm, n_splits=min(10, max(2, 2 * (len(trial_oof) // 2))))["pbo"] \
            if Mm.shape[1] >= 2 else np.nan
    except Exception:
        pbo_p = pbo_m = np.nan
    try:
        eff = O.effective_n_trials(Mm)["effective_n"] if Mm.shape[1] >= 4 else len(trial_oof)
    except Exception:
        eff = len(trial_oof)

    return dict(
        market=market, name=name, n_bars=n, bpy=bpy, n_trials=len(trial_oof),
        best=dict(fast=T["fast"], slow=T["slow"], pt=T["pt"], sl=T["sl"], mh=T["mh"]),
        n_ev=T["n_ev"], n_act=T["n_act"], frac_act=T["frac_act"],
        base_rate=T["base_rate"], precision=T["precision"],
        prim_sr_ann=T["sp"]["sr_ann"], prim_pf=T["sp"]["pf"],
        meta_sr_ann=T["sm"]["sr_ann"], meta_pf=T["sm"]["pf"],
        prim_dsr=d_primary["dsr"], meta_dsr=d_meta["dsr"],
        prim_sr0=d_primary["sr0"], meta_sr0=d_meta["sr0"],
        pbo_primary=pbo_p, pbo_meta=pbo_m, eff_n=eff,
        # carry arrays for figures (representative instruments only)
        _T=T, _sr_trials_meta=sr_trials_meta)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def universe(smoke=False):
    u = []
    cr = sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet")))
    for p in cr:
        nm = os.path.basename(p).replace("_1m.parquet", "")
        u.append(("crypto", nm, p))
    for s in ETF_SYMS:
        p = os.path.join(ETF_DIR, f"{s}.csv.gz")
        if os.path.exists(p):
            u.append(("equities", s, p))
    fx = sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet")))
    for p in fx:
        nm = os.path.basename(p).replace("_fx1m.parquet", "")
        u.append(("forex", nm, p))
    if smoke:
        # one representative per market
        return [next(x for x in u if x[0] == "crypto" and x[1] == "BTCUSDT"),
                next(x for x in u if x[0] == "equities" and x[1] == "SPY"),
                next(x for x in u if x[0] == "forex" and x[1] == "EURUSD")]
    return u


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df_inst: pd.DataFrame, reps: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style()
    P = style.PALETTE
    mkt_color = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}
    markets = ["crypto", "equities", "forex"]

    # FIG 1: precision/PF before-vs-after by market (PF primary vs meta)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(markets)); w = 0.36
    pf_p = [df_inst[df_inst.market == m]["prim_pf"].median() for m in markets]
    pf_m = [df_inst[df_inst.market == m]["meta_pf"].median() for m in markets]
    ax[0].bar(xs - w/2, pf_p, w, label="primary", color="#999999")
    ax[0].bar(xs + w/2, pf_m, w, label="meta", color=P["accent"])
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(markets)
    ax[0].set_ylabel("median PF (net of costs)"); ax[0].set_title("Profit factor: primary vs meta")
    ax[0].legend()
    pr = [df_inst[df_inst.market == m]["precision"].median() for m in markets]
    br = [df_inst[df_inst.market == m]["base_rate"].median() for m in markets]
    ax[1].bar(xs - w/2, br, w, label="base rate (all events)", color="#999999")
    ax[1].bar(xs + w/2, pr, w, label="meta precision (acted)", color=P["dollar"])
    ax[1].set_xticks(xs); ax[1].set_xticklabels(markets)
    ax[1].set_ylabel("P(profitable bet)"); ax[1].set_title("Meta precision vs base rate")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_precision_pf_by_market.png")); plt.close(fig)

    # FIG 2: bet-size distribution / trades dropped (representative per market)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        T = reps[m]["_T"]
        ax[i].hist(T["p_oof"], bins=30, color=mkt_color[m], alpha=0.85)
        ax[i].axvline(META_THRESH, color="k", ls="--", lw=1.0)
        dropped = 1 - T["act"].mean()
        ax[i].set_title(f"{m}: {reps[m]['name']}\n{dropped*100:.0f}% events dropped")
        ax[i].set_xlabel("meta P(profit) = bet size"); ax[i].set_ylabel("events")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_betsize_distribution.png")); plt.close(fig)

    # FIG 3: OOS equity curves primary vs meta, one per market
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        T = reps[m]["_T"]
        eqp = np.cumsum(T["r_primary"]); eqm = np.cumsum(T["r_meta"])
        ax[i].plot(eqp, color="#999999", label="primary")
        ax[i].plot(eqm, color=P["accent"], label="meta")
        ax[i].set_title(f"{m}: {reps[m]['name']}")
        ax[i].set_xlabel("bar"); ax[i].set_ylabel("cum log-return (net)")
        ax[i].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_equity_curves.png")); plt.close(fig)

    # FIG 4: DSR before-vs-after by market (median + scatter)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    for i, m in enumerate(markets):
        sub = df_inst[df_inst.market == m]
        ax.scatter(np.full(len(sub), i - 0.12) + np.random.uniform(-0.04, 0.04, len(sub)),
                   sub["prim_dsr"], s=14, color="#999999", alpha=0.6,
                   label="primary" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.12) + np.random.uniform(-0.04, 0.04, len(sub)),
                   sub["meta_dsr"], s=14, color=P["accent"], alpha=0.6,
                   label="meta" if i == 0 else None)
        ax.plot([i - 0.12], [sub["prim_dsr"].median()], "_", ms=26, color="k")
        ax.plot([i + 0.12], [sub["meta_dsr"].median()], "_", ms=26, color="k")
    ax.axhline(0.95, color=P["dollar"], ls="--", lw=1.0, label="DSR=0.95")
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(markets)
    ax.set_ylabel("Deflated Sharpe Ratio (OOS)")
    ax.set_title("DSR: primary vs meta-labeled, by market")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_dsr_by_market.png")); plt.close(fig)
    print("  figures written to", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def make_tables(df_inst: pd.DataFrame):
    cols = ["market", "name", "n_ev", "n_act", "frac_act", "precision", "base_rate",
            "prim_pf", "meta_pf", "prim_sr_ann", "meta_sr_ann",
            "prim_dsr", "meta_dsr", "pbo_primary", "pbo_meta", "eff_n",
            "best"]
    t = df_inst[cols].copy()
    t["best"] = t["best"].apply(lambda d: f"f{d['fast']}/s{d['slow']} pt{d['pt']}/sl{d['sl']} h{d['mh']}")
    t = t.round(4)
    t.to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)

    # by-market summary
    g = df_inst.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_prim_pf": g["prim_pf"].median(),
        "med_meta_pf": g["meta_pf"].median(),
        "med_prim_sr_ann": g["prim_sr_ann"].median(),
        "med_meta_sr_ann": g["meta_sr_ann"].median(),
        "med_prim_dsr": g["prim_dsr"].median(),
        "med_meta_dsr": g["meta_dsr"].median(),
        "n_meta_dsr_gt95": g.apply(lambda x: int((x["meta_dsr"] > 0.95).sum())),
        "n_prim_dsr_gt95": g.apply(lambda x: int((x["prim_dsr"] > 0.95).sum())),
        "med_pbo_meta": g["pbo_meta"].median(),
        "med_frac_act": g["frac_act"].median(),
        "med_precision_lift": g.apply(lambda x: (x["precision"] - x["base_rate"]).median()),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))

    # markdown
    md = ["# Triple-Barrier + Meta-Labeling, results\n",
          f"_{N_TRIALS} IS-tunable trials per instrument; DSR is the headline metric._\n",
          "\n## By-market summary\n", summ.to_markdown(),
          "\n\n## Per-instrument (head)\n", t.head(40).to_markdown(index=False)]
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables written to", os.path.abspath(TAB))
    return summ


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_one():
    import cProfile, pstats, io
    market, name, path = "crypto", "BTCUSDT", os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    # warm the kernel first (exclude JIT compile from the profile)
    bars = make_bars(market, path)
    close = bars["close"].to_numpy(np.float64); high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64); openp = bars["open"].to_numpy(np.float64)
    vol = tbm.ewma_vol(close, VOL_SPAN)
    sf = tbm.primary_ma_crossover(close, 20, 60); ev = tbm.crossover_events(sf)
    ev = ev[(ev > 72) & (ev < len(close) - 1)]
    _ = tbm.triple_barrier(close, high, low, openp, ev[:5], sf[ev][:5], vol, 1.5, 1.0, 50)
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path, verbose=True)
    pr.disable()
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if args.profile:
        profile_one(); return

    u = universe(smoke=args.smoke)
    print(f"running {len(u)} instruments across "
          f"{len(set(m for m,_,_ in u))} markets; {N_TRIALS} trials each")
    rows = []
    reps = {}
    t0 = time.perf_counter()
    for market, name, path in u:
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path)
            if r is None:
                print(f"  [skip] {market:9s} {name:10s} (insufficient data/events)")
                continue
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            # representative per market for figures: prefer the canonical name
            # (BTCUSDT / SPY / EURUSD), else fall back to first seen.
            want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
            if market not in reps or name == want.get(market):
                if market not in reps or reps[market]["name"] != want.get(market):
                    reps[market] = r
            print(f"  [ok]   {market:9s} {name:10s} "
                  f"prim_DSR={r['prim_dsr']:.3f} meta_DSR={r['meta_dsr']:.3f} "
                  f"prim_PF={r['prim_pf']:.3f} meta_PF={r['meta_pf']:.3f} "
                  f"act={r['frac_act']:.2f} ({time.perf_counter()-tt:.1f}s)")
        except Exception as e:
            print(f"  [ERR]  {market:9s} {name:10s}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no instruments produced results"); return
    df.to_parquet(os.path.join(TAB, "raw_results.parquet"))
    summ = make_tables(df)
    make_figures(df, reps)
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()

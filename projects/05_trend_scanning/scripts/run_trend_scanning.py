#!/usr/bin/env python3
"""
run_trend_scanning.py — Trend-Scanning labels at scale (LdP ML4AM Ch.5).

Idempotent driver. Reproduces LdP's trend-scanning labeller and tests, across
Crypto + US Equities + Forex, whether trend-scanning labels (sign of the max-|t|
forward OLS slope, with |t| as confidence) produce a better tradable side + size
signal than a FIXED-HORIZON labeller — judged by the program HEADLINE METRIC, the
Deflated Sharpe Ratio (not raw PF/Sharpe), net of realistic costs, with PBO +
effective-N, under PURGED K-FOLD CV (leakage-free).

Pipeline per instrument:
  1. Load real 1m base -> dollar bars (crypto/equities) or tick bars (forex).
  2. LABELS (forward-looking targets, NOT features):
       - trend-scan: for each bar, max-|t| forward OLS slope over L in [Lmin,Lmax];
         label = sign(t), confidence = |t|, winning horizon L*, realised window ret.
       - fixed-horizon control: OLS slope t-value at a single fixed L.
  3. EVENTS = bars where |t| (trend-scan) clears an IS-tuned confidence quantile
     (LdP samples on significant trends, not every bar) -> non/low-overlap events.
  4. SIDE = label sign. SIZE/META: a SECONDARY model predicts P(the labelled side
     is realised profitable net of costs) from CAUSAL features, trained & scored
     with PURGED K-FOLD CV (label_span = max horizon) -> leakage-free OOF score.
     Strategy acts only when OOF P>=thresh, sized by P (LdP size~prob).
  5. Compare TREND-SCAN side+|t|-size vs FIXED-HORIZON side, both costed, on the
     per-bar OOS return series.
  6. Score with annualised Sharpe, PF, and the DEFLATED SHARPE RATIO (the IS-tuned
     knob grid = the trial count -> multiple testing); report PBO + effective-N.

Causality: trend-scan / fixed-horizon are LABELS (supervised targets). The model
is trained on (causal-feature, label) pairs under purged CV and only ever acts on
OUT-OF-FOLD scores; no window-t-or-later information enters the features or the
acted-on score. Forward look-ahead lives only in the target, as LdP intends.

Knobs (FAST/SLOW windows, confidence quantile, horizon band, max-hold) are
IS-TUNABLE: a small grid is the trial set, scored OOF; the best-OOF trial is the
"shipped" winner and its SR is deflated against the trial dispersion.

Run:  python3 scripts/run_trend_scanning.py            (full multi-market run)
      python3 scripts/run_trend_scanning.py --smoke     (one instrument per market, tiny)
      python3 scripts/run_trend_scanning.py --profile   (cProfile a single instrument)
      python3 scripts/run_trend_scanning.py --verify    (kernel vs reference, max|d|)
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/daru/ldp_review"
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import trendscan as TS

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
CRYPTO_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_1m"
FX_DIR = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
ETF_DIR = "/mnt/d/algoseek_data/etf_1min"
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-side cost in bp of notional, applied on entry AND exit (full turnover = 2x).
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000           # dollar/tick bars per instrument (LdP-style)
VOL_SPAN = 50

# IS-tunable knob grid (the "trials"). Structural shape fixed (trend-scan label +
# bagged-tree meta). Numeric knobs tuned IN-SAMPLE (OOF-selected).
HORIZON_BANDS = [(5, 30), (10, 60), (20, 120)]   # (Lmin, Lmax) forward OLS window
CONF_Q = [0.60, 0.75, 0.90]                       # |t| quantile -> event filter
MAX_HOLD_FRAC = [1.0]                             # cap hold at L* (frac of L*)
# fixed-horizon control L = midpoint of each band (for apples-to-apples)
PARAM_GRID = [(hb, q) for hb in HORIZON_BANDS for q in CONF_Q]
N_TRIALS = len(PARAM_GRID)                         # = 9 trials

N_SPLITS = 6
EMBARGO = 0.01
META_THRESH = 0.50
RANDOM_STATE = 0

# smoke overrides (1-core, tiny): set in main()
SMOKE = dict(n_target=4000, grid=[((10, 60), 0.75)], n_splits=4, max_inst=None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_bars(market: str, path: str, n_target: int) -> pd.DataFrame:
    if market == "crypto":
        base = TS.load_base_crypto(path)
        return B.matched_bars(base, n_target)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True)
        return B.matched_bars(base, n_target)["dollar"]
    if market == "forex":
        base = TS.load_base_fx(path)
        thr = base["count"].sum() / n_target
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars: pd.DataFrame) -> float:
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def trade_returns_to_series(events_idx, hold, pnl_net, n_bars):
    """Spread each trade's net log-return across its holding bars -> per-bar return
    series for Sharpe/DSR. Overlapping bets sum (portfolio of concurrent bets)."""
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
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr * np.sqrt(bpy), pf=pf, skew=sk, kurt=ku,
                mean=float(r.mean()), n_obs=len(r))


# --------------------------------------------------------------------------- #
# Core: run one instrument
# --------------------------------------------------------------------------- #
def run_instrument(market, name, path, n_target, grid, n_splits, verbose=False):
    bars = make_bars(market, path, n_target)
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    n = len(close)
    # REALISTIC per-side cost: time-of-day half-spread (+commission for equity)
    # as a per-bar fraction for EQUITY/FOREX; crypto keeps the flat house default.
    # Causal (time-of-day known ex-ante). Round trip charges entry+exit per trade.
    cost = COST_BP[market] / 1e4   # retained for crypto / fallback references
    cost_side = RZ.per_side_cost_fraction(market, name, bars.index, close,
                                          crypto_fallback=cost)
    vol = TS.ewma_vol(close, VOL_SPAN)
    feat_full = TS.build_features(bars, vol)

    trial_rows = []
    sr_trials_ts, sr_trials_fx = [], []

    for (l_min, l_max), q in grid:
        # ----- LABELS (forward-looking targets) -----
        ts = TS.trend_scan(close, l_min, l_max)
        L_fixed = (l_min + l_max) // 2
        fh = TS.fixed_horizon(close, L_fixed)

        t_val = ts["t_val"].to_numpy()
        ls_star = ts["lstar"].to_numpy()
        ts_lab = ts["label"].to_numpy()
        ts_ret = ts["ret"].to_numpy()           # realised log-ret over winning win
        fh_lab = fh["label"].to_numpy()
        fh_ret = fh["ret"].to_numpy()

        warm = l_max + VOL_SPAN + 5
        abs_t = np.abs(t_val)
        valid = np.isfinite(abs_t)
        valid[:warm] = False
        if valid.sum() < 400:
            continue
        thr_conf = np.nanquantile(abs_t[valid], q)
        ev = np.where(valid & (abs_t >= thr_conf) & (ts_lab != 0))[0].astype(np.int64)
        if len(ev) < 200:
            continue

        # ----- CAUSAL execution & the leakage-free experiment -----
        # The trend-scan / fixed-horizon SIGNS are LABELS (forward-looking targets),
        # so we CANNOT trade them directly (that is the label leaking into P&L).
        # Instead a secondary classifier PREDICTS the label sign from CAUSAL
        # features under purged CV; the strategy trades the OUT-OF-FOLD predicted
        # side, entering at the event close and exiting after a fixed forward hold
        # (causally executable). We compare a model trained on TREND-SCAN labels vs
        # one trained on FIXED-HORIZON labels — the real question LdP poses: do
        # trend-scan labels make a better supervised target?
        hold_exec, hbar = TS.causal_hold_ret(close, ev, L_fixed)   # unsigned ret

        # binary side targets in {0,1} (1=up, 0=down); events are non-zero by filter
        y_ts = (ts_lab[ev] > 0).astype(int)
        y_fx = (fh_lab[ev] > 0).astype(int)

        # causal features at the event bars (no lookahead)
        X = feat_full.iloc[ev].to_numpy(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        def oof_predicted_side(y):
            """Purged-CV out-of-fold P(up); -> predicted side in {+1,-1} and a
            confidence |2p-1|. Degrades to the train-fold majority side when a fold
            is single-class. Returns (pred_side, conf, base_acc)."""
            p = np.full(len(ev), np.nan)
            for tr, te in O.purged_kfold_splits(len(ev), n_splits, EMBARGO, label_span=3):
                if len(tr) < 50:
                    p[te] = float(y[tr].mean()) if len(tr) else float(y.mean()); continue
                if y[tr].sum() < 5 or (1 - y[tr]).sum() < 5:
                    p[te] = float(y[tr].mean()); continue
                clf = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
                    n_estimators=40, max_samples=0.8, max_features=0.8,
                    bootstrap=True, n_jobs=1, random_state=RANDOM_STATE)
                clf.fit(X[tr], y[tr])
                if len(clf.classes_) == 1:
                    p[te] = float(clf.classes_[0])
                else:
                    pi = list(clf.classes_).index(1)
                    p[te] = clf.predict_proba(X[te])[:, pi]
            p = np.nan_to_num(p, nan=float(y.mean()))
            pred = np.where(p >= 0.5, 1, -1).astype(np.int64)
            conf = np.abs(2.0 * p - 1.0)
            acc = float((pred == np.where(y > 0, 1, -1)).mean())
            return pred, conf, acc

        pred_ts, conf_ts, acc_ts = oof_predicted_side(y_ts)
        pred_fx, conf_fx, acc_fx = oof_predicted_side(y_fx)

        # ----- strategy series: trade the OOF-PREDICTED side over a causal hold ---
        # trend-scan model: size ~ OOF confidence (LdP bet sizing). Act when conf>0.
        act_ts = conf_ts > 0.0
        size_ts = np.where(act_ts, conf_ts, 0.0)
        # per-trade realistic round-trip cost = entry-bar + exit-bar per-side cost
        exit_bar = np.minimum(ev + np.maximum(1, hbar.astype(np.int64)), n - 1)
        rt_cost = cost_side[ev] + cost_side[exit_bar]
        gross_ts = pred_ts * hold_exec
        pnl_ts = size_ts * gross_ts - rt_cost * size_ts
        r_ts = trade_returns_to_series(ev, hbar, pnl_ts, n)
        # fixed-horizon model control: unit size on predicted side (the plain target)
        gross_fx = pred_fx * hold_exec
        pnl_fx = gross_fx - rt_cost
        r_fx = trade_returns_to_series(ev, hbar, pnl_fx, n)

        s_ts = score_series(r_ts, bpy)
        s_fx = score_series(r_fx, bpy)
        sr_trials_ts.append(s_ts["sr_per_bar"])
        sr_trials_fx.append(s_fx["sr_per_bar"])
        trial_rows.append(dict(
            l_min=l_min, l_max=l_max, q=q, L_fixed=L_fixed,
            n_ev=len(ev), n_act=int(act_ts.sum()), frac_act=float(act_ts.mean()),
            base_rate=float(max(y_ts.mean(), 1 - y_ts.mean())),  # majority-side acc
            precision=acc_ts,                                     # OOF side accuracy
            med_lstar=float(np.median(ls_star[ev])),
            s_ts=s_ts, s_fx=s_fx, r_ts=r_ts, r_fx=r_fx,
            p_oof=conf_ts, act=act_ts, ev=ev, t_val_ev=t_val[ev]))

    if not trial_rows:
        return None
    sr_trials_ts = np.array(sr_trials_ts); sr_trials_fx = np.array(sr_trials_fx)
    best = int(np.argmax([t["s_ts"]["sr_per_bar"] for t in trial_rows]))
    T = trial_rows[best]

    d_ts = O.deflated_sharpe_ratio(T["s_ts"]["sr_per_bar"], T["s_ts"]["n_obs"],
                                   T["s_ts"]["skew"], T["s_ts"]["kurt"], sr_trials_ts)
    d_fx = O.deflated_sharpe_ratio(T["s_fx"]["sr_per_bar"], T["s_fx"]["n_obs"],
                                   T["s_fx"]["skew"], T["s_fx"]["kurt"], sr_trials_fx)
    Mts = np.column_stack([t["r_ts"] for t in trial_rows])
    Mfx = np.column_stack([t["r_fx"] for t in trial_rows])
    try:
        ns = min(10, max(2, 2 * (len(trial_rows) // 2)))
        pbo_ts = O.pbo_cscv(Mts, n_splits=ns)["pbo"] if Mts.shape[1] >= 2 else np.nan
        pbo_fx = O.pbo_cscv(Mfx, n_splits=ns)["pbo"] if Mfx.shape[1] >= 2 else np.nan
    except Exception:
        pbo_ts = pbo_fx = np.nan
    try:
        eff = O.effective_n_trials(Mts)["effective_n"] if Mts.shape[1] >= 4 else len(trial_rows)
    except Exception:
        eff = len(trial_rows)

    return dict(
        market=market, name=name, n_bars=n, bpy=bpy, n_trials=len(trial_rows),
        best=dict(l_min=T["l_min"], l_max=T["l_max"], q=T["q"], L_fixed=T["L_fixed"]),
        n_ev=T["n_ev"], n_act=T["n_act"], frac_act=T["frac_act"],
        base_rate=T["base_rate"], precision=T["precision"], med_lstar=T["med_lstar"],
        ts_sr_ann=T["s_ts"]["sr_ann"], ts_pf=T["s_ts"]["pf"],
        fx_sr_ann=T["s_fx"]["sr_ann"], fx_pf=T["s_fx"]["pf"],
        ts_dsr=d_ts["dsr"], fx_dsr=d_fx["dsr"], ts_sr0=d_ts["sr0"], fx_sr0=d_fx["sr0"],
        pbo_ts=pbo_ts, pbo_fx=pbo_fx, eff_n=eff,
        _T=T)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def universe(smoke=False):
    u = []
    for p in sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet"))):
        u.append(("crypto", os.path.basename(p).replace("_1m.parquet", ""), p))
    for s in ETF_SYMS:
        p = os.path.join(ETF_DIR, f"{s}.csv.gz")
        if os.path.exists(p):
            u.append(("equities", s, p))
    for p in sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet"))):
        u.append(("forex", os.path.basename(p).replace("_fx1m.parquet", ""), p))
    if smoke:
        pick = lambda mk, nm: next(x for x in u if x[0] == mk and x[1] == nm)
        return [pick("crypto", "BTCUSDT"), pick("equities", "SPY"), pick("forex", "EURUSD")]
    return u


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df_inst, reps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    mkt_color = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}
    markets = ["crypto", "equities", "forex"]

    # FIG 1: PF & DSR trend-scan vs fixed-horizon by market
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(markets)); w = 0.36
    pf_t = [df_inst[df_inst.market == m]["ts_pf"].median() for m in markets]
    pf_f = [df_inst[df_inst.market == m]["fx_pf"].median() for m in markets]
    ax[0].bar(xs - w/2, pf_f, w, label="fixed-horizon", color="#999999")
    ax[0].bar(xs + w/2, pf_t, w, label="trend-scan", color=P["accent"])
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(markets)
    ax[0].set_ylabel("median PF (net of costs)")
    ax[0].set_title("Profit factor: fixed-horizon vs trend-scan"); ax[0].legend()
    d_t = [df_inst[df_inst.market == m]["ts_dsr"].median() for m in markets]
    d_f = [df_inst[df_inst.market == m]["fx_dsr"].median() for m in markets]
    ax[1].bar(xs - w/2, d_f, w, label="fixed-horizon", color="#999999")
    ax[1].bar(xs + w/2, d_t, w, label="trend-scan", color=P["accent"])
    ax[1].axhline(0.95, color=P["dollar"], lw=0.9, ls="--", label="DSR=0.95")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(markets)
    ax[1].set_ylabel("median Deflated Sharpe Ratio")
    ax[1].set_title("DSR: fixed-horizon vs trend-scan"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_pf_dsr_by_market.png")); plt.close(fig)

    # FIG 2: |t| (confidence) distribution at events, representative per market
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        T = reps[m]["_T"]
        ax[i].hist(np.abs(T["t_val_ev"]), bins=30, color=mkt_color[m], alpha=0.85)
        ax[i].set_title(f"{m}: {reps[m]['name']}\nmedian L*={reps[m]['med_lstar']:.0f} bars")
        ax[i].set_xlabel("|t| of max-slope (trend-scan confidence)"); ax[i].set_ylabel("events")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_tval_confidence.png")); plt.close(fig)

    # FIG 3: OOS equity curves trend-scan vs fixed-horizon, per market
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, m in enumerate(markets):
        T = reps[m]["_T"]
        ax[i].plot(np.cumsum(T["r_fx"]), color="#999999", label="fixed-horizon")
        ax[i].plot(np.cumsum(T["r_ts"]), color=P["accent"], label="trend-scan")
        ax[i].set_title(f"{m}: {reps[m]['name']}")
        ax[i].set_xlabel("bar"); ax[i].set_ylabel("cum log-return (net)"); ax[i].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_equity_curves.png")); plt.close(fig)

    # FIG 4: DSR scatter by market
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    rng = np.random.default_rng(0)
    for i, m in enumerate(markets):
        sub = df_inst[df_inst.market == m]
        ax.scatter(np.full(len(sub), i - 0.12) + rng.uniform(-0.04, 0.04, len(sub)),
                   sub["fx_dsr"], s=14, color="#999999", alpha=0.6,
                   label="fixed-horizon" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.12) + rng.uniform(-0.04, 0.04, len(sub)),
                   sub["ts_dsr"], s=14, color=P["accent"], alpha=0.6,
                   label="trend-scan" if i == 0 else None)
        ax.plot([i - 0.12], [sub["fx_dsr"].median()], "_", ms=26, color="k")
        ax.plot([i + 0.12], [sub["ts_dsr"].median()], "_", ms=26, color="k")
    ax.axhline(0.95, color=P["dollar"], ls="--", lw=1.0, label="DSR=0.95")
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(markets)
    ax.set_ylabel("Deflated Sharpe Ratio (OOS)")
    ax.set_title("DSR per instrument: trend-scan vs fixed-horizon"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_dsr_scatter.png")); plt.close(fig)
    print("  figures written to", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def make_tables(df_inst):
    cols = ["market", "name", "n_ev", "n_act", "frac_act", "precision", "base_rate",
            "med_lstar", "ts_pf", "fx_pf", "ts_sr_ann", "fx_sr_ann",
            "ts_dsr", "fx_dsr", "pbo_ts", "pbo_fx", "eff_n", "best"]
    t = df_inst[cols].copy()
    t["best"] = t["best"].apply(lambda d: f"L[{d['l_min']},{d['l_max']}] q{d['q']} Lfix{d['L_fixed']}")
    t = t.round(4)
    t.to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)

    g = df_inst.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_ts_pf": g["ts_pf"].median(), "med_fx_pf": g["fx_pf"].median(),
        "med_ts_sr_ann": g["ts_sr_ann"].median(), "med_fx_sr_ann": g["fx_sr_ann"].median(),
        "med_ts_dsr": g["ts_dsr"].median(), "med_fx_dsr": g["fx_dsr"].median(),
        "n_ts_dsr_gt95": g.apply(lambda x: int((x["ts_dsr"] > 0.95).sum())),
        "n_fx_dsr_gt95": g.apply(lambda x: int((x["fx_dsr"] > 0.95).sum())),
        "med_pbo_ts": g["pbo_ts"].median(),
        "med_frac_act": g["frac_act"].median(),
        "med_precision_lift": g.apply(lambda x: (x["precision"] - x["base_rate"]).median()),
        "med_lstar": g["med_lstar"].median(),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))
    md = ["# Trend-Scanning labels vs fixed-horizon — results\n",
          f"_{len(PARAM_GRID)} IS-tunable trials per instrument; DSR is the headline metric._\n",
          "\n## By-market summary\n", summ.to_markdown(),
          "\n\n## Per-instrument (head)\n", t.head(50).to_markdown(index=False)]
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables written to", os.path.abspath(TAB))
    return summ


# --------------------------------------------------------------------------- #
# Verify / Profile
# --------------------------------------------------------------------------- #
def verify():
    rng = np.random.default_rng(7)
    print("=== trend-scan kernel vs independent NumPy reference ===")
    worst_t = worst_r = 0.0
    lab_ok = ls_ok = True
    for seed in (1, 7, 42, 99):
        rng = np.random.default_rng(seed)
        px = 100 * np.exp(np.cumsum(rng.standard_normal(1000) * 0.01))
        A = TS.trend_scan(px, 5, 40); Bf = TS.trend_scan_reference(px, 5, 40)
        a, b = A["t_val"].to_numpy(), Bf["t_val"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        worst_t = max(worst_t, float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0)
        ar, br = A["ret"].to_numpy(), Bf["ret"].to_numpy()
        mr = np.isfinite(ar) & np.isfinite(br)
        worst_r = max(worst_r, float(np.max(np.abs(ar[mr] - br[mr]))) if mr.any() else 0.0)
        lab_ok &= np.array_equal(A["label"].to_numpy(), Bf["label"].to_numpy())
        ls_ok &= np.array_equal(A["lstar"].to_numpy(), Bf["lstar"].to_numpy())
    print(f"  max|d| t_val = {worst_t:.3e}   max|d| ret = {worst_r:.3e}")
    print(f"  label bit-identical = {lab_ok}   lstar bit-identical = {ls_ok}")
    print("  (t_val delta is float summation-order only; label/lstar/ret exact -> "
          "P&L bit-identical)")


def profile_one():
    import cProfile, pstats, io
    market, name = "crypto", "BTCUSDT"
    path = os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    bars = make_bars(market, path, N_TARGET_BARS)
    close = bars["close"].to_numpy(np.float64)
    _ = TS.trend_scan(close[:2000], 5, 30)            # warm JIT (exclude compile)
    _ = TS.fixed_horizon(close[:2000], 20)
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path, N_TARGET_BARS, PARAM_GRID, N_SPLITS, verbose=True)
    pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        verify(); return
    if args.profile:
        profile_one(); return

    if args.smoke:
        n_target = SMOKE["n_target"]; grid = SMOKE["grid"]; n_splits = SMOKE["n_splits"]
    else:
        n_target = N_TARGET_BARS; grid = PARAM_GRID; n_splits = N_SPLITS

    u = universe(smoke=args.smoke)
    print(f"running {len(u)} instruments across {len(set(m for m,_,_ in u))} markets; "
          f"{len(grid)} trials each; n_target_bars={n_target}")
    rows, reps = [], {}
    want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
    t0 = time.perf_counter()
    for market, name, path in u:
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path, n_target, grid, n_splits)
            if r is None:
                print(f"  [skip] {market:9s} {name:10s} (insufficient data/events)")
                continue
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            if market not in reps or name == want.get(market):
                reps[market] = r
            print(f"  [ok]   {market:9s} {name:10s} "
                  f"ts_DSR={r['ts_dsr']:.3f} fx_DSR={r['fx_dsr']:.3f} "
                  f"ts_PF={r['ts_pf']:.3f} fx_PF={r['fx_pf']:.3f} "
                  f"act={r['frac_act']:.2f} L*={r['med_lstar']:.0f} "
                  f"({time.perf_counter()-tt:.1f}s)")
        except Exception as e:
            print(f"  [ERR]  {market:9s} {name:10s}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no instruments produced results"); return
    # flatten the 'best' dict column so the frame is parquet-serialisable
    df_pq = df.copy()
    if "best" in df_pq.columns:
        df_pq["best"] = df_pq["best"].apply(
            lambda d: f"L[{d['l_min']},{d['l_max']}] q{d['q']} Lfix{d['L_fixed']}")
    df_pq.to_parquet(os.path.join(TAB, "raw_results.parquet"))
    summ = make_tables(df)
    for m in ("crypto", "equities", "forex"):
        if m not in reps:
            reps[m] = next(r for r in [run_instrument(*x, n_target, grid, n_splits)
                                       for x in u if x[0] == m] if r)
    try:
        make_figures(df, reps)
    except Exception as e:
        print(f"  [fig ERR] {e}")
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()

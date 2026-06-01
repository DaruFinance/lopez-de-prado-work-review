#!/usr/bin/env python3
"""
deepen_labels.py, deepening pass for Project 05 (Trend-Scanning labels).

The cleanest comparison LdP poses in ML4AM Ch.5: holding the *secondary model*,
the *events*, the *causal features*, the *purged CV*, the *costs*, and the
*causal hold execution* all FIXED, swap ONLY the LABELLING TARGET and ask which
target produces a side+size signal that survives deflation OOS, by market:

    target A : TREND-SCAN     sign(max-|t| forward OLS slope)          [Ch.5 method]
    target B : FIXED-HORIZON  sign(OLS t at a single L)                [the control]
    target C : TRIPLE-BARRIER sign(first-touch outcome of a primary)   [Ch.3 labeller]

For C the primary side at each event is the trend-scan label sign (so all three
targets sit on the SAME event set); the TBM label is the realised first-touch
sign using full intrabar OHLC (pessimistic ordering). The secondary classifier
predicts each target out-of-fold; the strategy trades the OOF-predicted side over
the SAME causal forward hold L_fixed, costed identically with lib.realism.

Two outputs:
  (1) by-market A vs B vs C: median DSR, #DSR>0.95, OOS-Sharpe, precision-lift.
  (2) HORIZON-BAND ROBUSTNESS: for the trend-scan target, DSR of EACH band
      (5,30)/(10,60)/(20,120) at a fixed quantile, per market, does the edge
      depend on a single hand-picked look-forward window, or hold across the band?

CAUSAL: forward look-ahead lives only in the label. Features, events' acted side,
and the hold exit use no window-t-or-later info. REALISTIC costs (no clamping).
DSR is the headline. Profiled 1-core; the new TBM kernel is verified bit-identical
against its pure-Python reference in this driver (--verify) before any run.

Run:  python3 scripts/deepen_labels.py --verify
      python3 scripts/deepen_labels.py --smoke
      python3 scripts/deepen_labels.py            (full 42-instrument, 1 worker/inst)
      python3 scripts/deepen_labels.py --jobs 6   (parallel across instruments)
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob, json
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "projects", "03_meta_labeling", "scripts"))

import numpy as np
import pandas as pd
from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import trendscan as TS
import tbm as TBM

from scipy import stats as ss
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

TAB = os.path.join(HERE, "..", "tables")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ---- config mirrors run_trend_scanning.py exactly so results are comparable ----
CRYPTO_DIR = cfg.CRYPTO_1M
FX_DIR = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000
VOL_SPAN = 50
HORIZON_BANDS = [(5, 30), (10, 60), (20, 120)]
CONF_Q = [0.60, 0.75, 0.90]
PARAM_GRID = [(hb, q) for hb in HORIZON_BANDS for q in CONF_Q]
N_SPLITS = 6
EMBARGO = 0.01
RANDOM_STATE = 0
# triple-barrier knobs (symmetric, vol-scaled), held fixed across the grid
TBM_PT, TBM_SL = 2.0, 2.0


def make_bars(market, path, n_target):
    if market == "crypto":
        base = TS.load_base_crypto(path); return B.matched_bars(base, n_target)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True); return B.matched_bars(base, n_target)["dollar"]
    if market == "forex":
        base = TS.load_base_fx(path)
        thr = base["count"].sum() / n_target
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars):
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    return len(bars) / max(span_s / (365.25 * 24 * 3600), 1e-6)


def trade_returns_to_series(events_idx, hold, pnl_net, n_bars):
    r = np.zeros(n_bars)
    for k in range(len(events_idx)):
        i0 = int(events_idx[k]); h = max(1, int(hold[k]))
        per = pnl_net[k] / h
        r[i0:min(i0 + h, n_bars)] += per
    return r


def score_series(r, bpy):
    r = r[np.isfinite(r)]; nz = r[r != 0.0]
    sr = O.sharpe(r)
    sk = float(ss.skew(r)) if len(r) > 2 else 0.0
    ku = float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else np.nan
    return dict(sr_per_bar=sr, sr_ann=sr * np.sqrt(bpy), pf=pf, skew=sk, kurt=ku, n_obs=len(r))


def oof_predicted_side(X, y, n_ev, n_splits, label_span):
    """Purged-CV OOF P(up) -> predicted side {+1,-1}, conf |2p-1|, OOF accuracy."""
    p = np.full(n_ev, np.nan)
    for tr, te in O.purged_kfold_splits(n_ev, n_splits, EMBARGO, label_span=label_span):
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
            p[te] = clf.predict_proba(X[te])[: pi]
    p = np.nan_to_num(p, nan=float(y.mean()))
    pred = np.where(p >= 0.5, 1, -1).astype(np.int64)
    conf = np.abs(2.0 * p - 1.0)
    acc = float((pred == np.where(y > 0, 1, -1)).mean())
    return pred, conf, acc


def run_instrument(market, name, path, n_target=N_TARGET_BARS, grid=PARAM_GRID,
                   n_splits=N_SPLITS):
    bars = make_bars(market, path, n_target)
    if len(bars) < 3000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    cost = COST_BP[market] / 1e4
    cost_side = RZ.per_side_cost_fraction(market, name, bars.index, close, crypto_fallback=cost)
    vol = TS.ewma_vol(close, VOL_SPAN)
    feat_full = TS.build_features(bars, vol)

    # per-trial SR streams for DSR deflation, per target
    sr_tr = {"ts": [], "fx": [], "tb": []}
    rows = []          # one per (band,q) trial
    band_rows = []     # for horizon-band robustness (trend-scan target only)

    for (l_min, l_max), q in grid:
        ts = TS.trend_scan(close, l_min, l_max)
        L_fixed = (l_min + l_max) // 2
        fh = TS.fixed_horizon(close, L_fixed)
        t_val = ts["t_val"].to_numpy(); ls_star = ts["lstar"].to_numpy()
        ts_lab = ts["label"].to_numpy(); fh_lab = fh["label"].to_numpy()

        warm = l_max + VOL_SPAN + 5
        abs_t = np.abs(t_val); valid = np.isfinite(abs_t); valid[:warm] = False
        if valid.sum() < 400:
            continue
        thr = np.nanquantile(abs_t[valid], q)
        ev = np.where(valid & (abs_t >= thr) & (ts_lab != 0))[0].astype(np.int64)
        if len(ev) < 200:
            continue

        # causal forward-hold execution (same for all three targets)
        hold_exec, hbar = TS.causal_hold_ret(close, ev, L_fixed)
        exit_bar = np.minimum(ev + np.maximum(1, hbar.astype(np.int64)), n - 1)
        rt_cost = cost_side[ev] + cost_side[exit_bar]

        X = np.nan_to_num(feat_full.iloc[ev].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # ---- target A: trend-scan side ----
        y_ts = (ts_lab[ev] > 0).astype(int)
        # ---- target B: fixed-horizon side ----
        y_fx = (fh_lab[ev] > 0).astype(int)
        # ---- target C: triple-barrier outcome sign, primary side = trend-scan ----
        tb = TBM.triple_barrier(close, high, low, openp, ev, ts_lab[ev], vol,
                                TBM_PT, TBM_SL, L_fixed)
        tb_lab_raw = tb["label"].to_numpy()      # +1 pt / -1 sl / 0 vertical
        # binary target = was the (trend-scan-sided) bet profitable at first touch?
        y_tb = (tb["ret_gross"].to_numpy() > 0).astype(int)

        pred_ts, conf_ts, acc_ts = oof_predicted_side(X, y_ts, len(ev), n_splits, 3)
        pred_fx, conf_fx, acc_fx = oof_predicted_side(X, y_fx, len(ev), n_splits, 3)
        # TBM as a META-LABEL overlay (LdP Ch.3/Ch.5): the SIDE is the SAME causal
        # OOF-predicted trend-scan side as target A (NEVER the forward-looking label
        # itself, that would leak the future into P&L). The triple-barrier outcome
        # is the supervised META target: a second OOF classifier predicts P(this
        # OOF side's bet survives the first-touch barriers profitably); we ACT/size
        # on that meta-probability. So target C answers: does a triple-barrier
        # *meta-filter* improve the trend-scan side vs trading it raw (A)?
        pred_tb_act, conf_tb, acc_tb = oof_predicted_side(X, y_tb, len(ev), n_splits, L_fixed)
        act_tb = pred_tb_act > 0                       # act when predicted-profitable
        side_tb = pred_ts                              # CAUSAL OOF side (== target A)

        gross_ts = pred_ts * hold_exec
        size_ts = conf_ts
        pnl_ts = size_ts * gross_ts - rt_cost * size_ts
        r_ts = trade_returns_to_series(ev, hbar, pnl_ts, n)

        gross_fx = pred_fx * hold_exec
        pnl_fx = gross_fx - rt_cost
        r_fx = trade_returns_to_series(ev, hbar, pnl_fx, n)

        size_tb = np.where(act_tb, conf_tb, 0.0)       # gate+size by meta-prob
        gross_tb = side_tb * hold_exec
        pnl_tb = size_tb * gross_tb - rt_cost * size_tb
        r_tb = trade_returns_to_series(ev, hbar, pnl_tb, n)

        s_ts = score_series(r_ts, bpy); s_fx = score_series(r_fx, bpy); s_tb = score_series(r_tb, bpy)
        sr_tr["ts"].append(s_ts["sr_per_bar"]); sr_tr["fx"].append(s_fx["sr_per_bar"]); sr_tr["tb"].append(s_tb["sr_per_bar"])
        rows.append(dict(l_min=l_min, l_max=l_max, q=q, n_ev=len(ev),
                         frac_act_tb=float(act_tb.mean()),
                         prec_ts=acc_ts, prec_fx=acc_fx, prec_tb=acc_tb,
                         base_ts=float(max(y_ts.mean(), 1 - y_ts.mean())),
                         base_tb=float(max(y_tb.mean(), 1 - y_tb.mean())),
                         tb_pt_rate=float((tb_lab_raw == 1).mean()),
                         tb_vert_rate=float((tb_lab_raw == 0).mean()),
                         med_lstar=float(np.median(ls_star[ev])),
                         s_ts=s_ts, s_fx=s_fx, s_tb=s_tb,
                         r_ts=r_ts, r_fx=r_fx, r_tb=r_tb))
        band_rows.append(dict(band=f"({l_min},{l_max})", q=q, sr_ts=s_ts["sr_per_bar"],
                              sr_ann=s_ts["sr_ann"], pf=s_ts["pf"], n_obs=s_ts["n_obs"],
                              skew=s_ts["skew"], kurt=s_ts["kurt"]))

    if not rows:
        return None

    out = dict(market=market, name=name, n_bars=n, bpy=bpy, n_trials=len(rows))
    # headline = best-OOF trend-scan trial, deflated against the per-target trial dispersion
    for tgt in ("ts", "fx", "tb"):
        srt = np.array(sr_tr[tgt])
        best = int(np.argmax([r[f"s_{tgt}"]["sr_per_bar"] for r in rows]))
        Rb = rows[best][f"s_{tgt}"]
        d = O.deflated_sharpe_ratio(Rb["sr_per_bar"], Rb["n_obs"], Rb["skew"], Rb["kurt"], srt)
        out[f"{tgt}_sr_ann"] = Rb["sr_ann"]; out[f"{tgt}_pf"] = Rb["pf"]
        out[f"{tgt}_dsr"] = d["dsr"]; out[f"{tgt}_sr0"] = d["sr0"]
        out[f"{tgt}_prec"] = rows[best][f"prec_{tgt}"]
        M = np.column_stack([r[f"r_{tgt}"] for r in rows])
        try:
            ns = min(10, max(2, 2 * (len(rows) // 2)))
            out[f"{tgt}_pbo"] = O.pbo_cscv(M, n_splits=ns)["pbo"] if M.shape[1] >= 2 else np.nan
        except Exception:
            out[f"{tgt}_pbo"] = np.nan
    out["base_ts"] = rows[best]["base_ts"]
    out["med_lstar"] = rows[best]["med_lstar"]
    out["tb_pt_rate"] = float(np.mean([r["tb_pt_rate"] for r in rows]))
    out["tb_vert_rate"] = float(np.mean([r["tb_vert_rate"] for r in rows]))
    out["frac_act_tb"] = float(np.mean([r["frac_act_tb"] for r in rows]))

    # ---- horizon-band robustness: DSR of trend-scan per band (deflate within band
    #      over its 3 quantile trials) ----
    bdf = pd.DataFrame(band_rows)
    band_out = {}
    for band, sub in bdf.groupby("band"):
        srt = sub["sr_ts"].to_numpy()
        bi = int(np.argmax(srt)); rb = sub.iloc[bi]
        d = O.deflated_sharpe_ratio(rb["sr_ts"], int(rb["n_obs"]), rb["skew"], rb["kurt"], srt)
        band_out[band] = dict(sr_ann=float(rb["sr_ann"]), pf=float(rb["pf"]), dsr=float(d["dsr"]))
    out["_band"] = band_out
    return out


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


def verify():
    """TBM kernel vs pure-Python reference on a trend-scan event set (bit-identical)."""
    rng = np.random.default_rng(11)
    px = 100 * np.exp(np.cumsum(rng.standard_normal(4000) * 0.01))
    hi = px * (1 + np.abs(rng.standard_normal(4000)) * 0.002)
    lo = px * (1 - np.abs(rng.standard_normal(4000)) * 0.002)
    op = px.copy()
    ts = TS.trend_scan(px, 10, 60)
    lab = ts["label"].to_numpy(); tv = np.abs(ts["t_val"].to_numpy())
    valid = np.isfinite(tv); valid[:120] = False
    ev = np.where(valid & (tv >= np.nanquantile(tv[valid], 0.75)) & (lab != 0))[0].astype(np.int64)
    vol = TS.ewma_vol(px, 50)
    A = TBM.triple_barrier(px, hi, lo, op, ev, lab[ev], vol, 2.0, 2.0, 35)
    Bf = TBM.triple_barrier_reference(px, hi, lo, ev, lab[ev], vol, 2.0, 2.0, 35)
    print("=== TBM kernel vs pure-Python reference (on trend-scan events) ===")
    for c in ("touch", "label", "hold"):
        print(f"  {c:6s} bit-identical = {np.array_equal(A[c].to_numpy(), Bf[c].to_numpy())}")
    dr = float(np.max(np.abs(A['ret_gross'].to_numpy() - Bf['ret_gross'].to_numpy())))
    print(f"  ret_gross max|d| = {dr:.3e}  (n_events={len(ev)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.verify:
        verify(); return

    grid = [((10, 60), 0.75)] if args.smoke else PARAM_GRID
    n_target = 6000 if args.smoke else N_TARGET_BARS
    n_splits = 4 if args.smoke else N_SPLITS
    u = universe(smoke=args.smoke)
    print(f"DEEPEN: {len(u)} instruments; targets=trend-scan/fixed-horizon/triple-barrier; "
          f"{len(grid)} trials; jobs={args.jobs}")
    t0 = time.perf_counter()
    rows, bands = [], []

    def collect(r):
        if r is None:
            return
        band = r.pop("_band")
        for b, d in band.items():
            bands.append(dict(market=r["market"], name=r["name"], band=b, **d))
        rows.append(r)
        print(f"  [ok] {r['market']:9s} {r['name']:10s} "
              f"ts_DSR={r['ts_dsr']:.3f} fx_DSR={r['fx_dsr']:.3f} tb_DSR={r['tb_dsr']:.3f} "
              f"ts_PF={r['ts_pf']:.2f} tb_PF={r['tb_pf']:.2f} actTB={r['frac_act_tb']:.2f}")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_instrument, m, nm, p, n_target, grid, n_splits): (m, nm)
                    for m, nm, p in u}
            for f in as_completed(futs):
                m, nm = futs[f]
                try:
                    collect(f.result())
                except Exception as e:
                    print(f"  [ERR] {m} {nm}: {e}")
    else:
        for m, nm, p in u:
            try:
                collect(run_instrument(m, nm, p, n_target, grid, n_splits))
            except Exception as e:
                print(f"  [ERR] {m} {nm}: {e}")

    if not rows:
        print("no results"); return
    df = pd.DataFrame(rows).sort_values(["market", "name"]).reset_index(drop=True)
    bdf = pd.DataFrame(bands)

    # ---- table 1: per-instrument 3-target ----
    keep = ["market", "name", "n_trials", "med_lstar", "frac_act_tb", "tb_pt_rate", "tb_vert_rate",
            "ts_prec", "fx_prec", "tb_prec", "base_ts",
            "ts_pf", "fx_pf", "tb_pf", "ts_sr_ann", "fx_sr_ann", "tb_sr_ann",
            "ts_dsr", "fx_dsr", "tb_dsr", "ts_pbo", "fx_pbo", "tb_pbo"]
    df[keep].round(4).to_csv(os.path.join(TAB, "deepen_per_instrument.csv"), index=False)

    # ---- table 2: by-market 3-target summary ----
    g = df.groupby("market")
    def med(c): return g[c].median()
    def gt95(c): return g.apply(lambda x: int((x[c] > 0.95).sum()))
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_ts_dsr": med("ts_dsr"), "med_fx_dsr": med("fx_dsr"), "med_tb_dsr": med("tb_dsr"),
        "n_ts_dsr95": gt95("ts_dsr"), "n_fx_dsr95": gt95("fx_dsr"), "n_tb_dsr95": gt95("tb_dsr"),
        "med_ts_pf": med("ts_pf"), "med_fx_pf": med("fx_pf"), "med_tb_pf": med("tb_pf"),
        "med_ts_sr_ann": med("ts_sr_ann"), "med_tb_sr_ann": med("tb_sr_ann"),
        "med_ts_pbo": med("ts_pbo"), "med_tb_pbo": med("tb_pbo"),
        "med_tb_pt_rate": med("tb_pt_rate"), "med_tb_vert": med("tb_vert_rate"),
        "med_frac_act_tb": med("frac_act_tb"),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "deepen_by_market.csv"))

    # ---- table 3: horizon-band robustness (trend-scan) ----
    gb = bdf.groupby(["market", "band"])
    bsum = pd.DataFrame({
        "n_inst": gb.size(),
        "med_dsr": gb["dsr"].median(),
        "n_dsr95": gb.apply(lambda x: int((x["dsr"] > 0.95).sum())),
        "med_pf": gb["pf"].median(),
        "med_sr_ann": gb["sr_ann"].median(),
    }).round(4)
    bsum.to_csv(os.path.join(TAB, "deepen_band_robustness.csv"))

    # ---- markdown ----
    md = ["# Deepening: trend-scan vs fixed-horizon vs triple-barrier (same secondary model)\n",
          f"_42 instruments; {len(PARAM_GRID)} IS-tunable trials; DSR headline; realistic costs._\n",
          "\n## By-market 3-target summary\n", summ.to_markdown(),
          "\n\n## Horizon-band robustness (trend-scan target; DSR per band)\n", bsum.to_markdown(),
          "\n\n## Per-instrument (head)\n", df[keep].round(3).head(50).to_markdown(index=False)]
    with open(os.path.join(TAB, "deepen_results.md"), "w") as f:
        f.write("\n".join(md))

    # save raw for the writeup
    df_pq = df.copy()
    df_pq.to_parquet(os.path.join(TAB, "deepen_raw.parquet"))
    bdf.to_parquet(os.path.join(TAB, "deepen_band_raw.parquet"))

    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET 3-TARGET ===")
    print(summ.to_string())
    print("\n=== HORIZON-BAND ROBUSTNESS (trend-scan) ===")
    print(bsum.to_string())

    make_figures(df, bdf)


def make_figures(df, bdf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    markets = ["crypto", "equities", "forex"]

    # FIG 5: 3-target median DSR by market
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xs = np.arange(len(markets)); w = 0.26
    for j, (tgt, lab, col) in enumerate([("fx", "fixed-horizon", "#999999"),
                                          ("ts", "trend-scan", P["accent"]),
                                          ("tb", "triple-barrier", P["dollar"])]):
        vals = [df[df.market == m][f"{tgt}_dsr"].median() for m in markets]
        ax[0].bar(xs + (j - 1) * w, vals, w, label=lab, color=col)
    ax[0].axhline(0.95, color="k", ls="--", lw=0.8)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(markets); ax[0].set_ylabel("median DSR")
    ax[0].set_title("Deflated Sharpe by labelling target"); ax[0].legend(fontsize=8)
    for j, (tgt, lab, col) in enumerate([("fx", "fixed-horizon", "#999999"),
                                          ("ts", "trend-scan", P["accent"]),
                                          ("tb", "triple-barrier", P["dollar"])]):
        vals = [int((df[df.market == m][f"{tgt}_dsr"] > 0.95).sum()) for m in markets]
        ax[1].bar(xs + (j - 1) * w, vals, w, label=lab, color=col)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(markets); ax[1].set_ylabel("# instruments DSR>0.95")
    ax[1].set_title("Count surviving deflation (DSR>0.95)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_three_target_dsr.png")); plt.close(fig)

    # FIG 6: horizon-band robustness heatmap-ish (median DSR per market x band)
    bands = ["(5,30)", "(10,60)", "(20,120)"]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    for i, band in enumerate(bands):
        vals = [bdf[(bdf.market == m) & (bdf.band == band)]["dsr"].median() for m in markets]
        ax.plot(markets, vals, "o-", label=f"L band {band}")
    ax.axhline(0.95, color="k", ls="--", lw=0.8)
    ax.set_ylabel("median trend-scan DSR"); ax.set_title("Trend-scan DSR across look-forward bands")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_band_robustness.png")); plt.close(fig)
    print("  figures: fig5_three_target_dsr.png, fig6_band_robustness.png")


if __name__ == "__main__":
    main()

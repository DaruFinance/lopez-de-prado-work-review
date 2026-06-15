#!/usr/bin/env python3
"""
s10_fx_only.py — run the 8 FX instruments that were skipped due to a
market-string mismatch ('forex' vs 'fx') in the original run.
Appends to s10_per_instrument.parquet and writes final s10_summary.json.
"""
from __future__ import annotations
import os, sys, gc, time, json, warnings, traceback, resource

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, FX_1M
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

from lib import bars as B
import overfit as O
import tbm
import uniqueness as U
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, roc_auc_score

OUTDIR   = HERE
FX_DIR   = FX_1M
FX_SYMS  = ["AUDUSD", "EURGBP", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

N_TARGET    = 20_000
N_EST       = 40
N_SPLITS    = 6
EMBARGO     = 0.01
LABEL_SPAN  = 3
SEED        = 0
FAST, SLOW  = 20, 60
PT, SL      = 1.5, 1.0
MAX_HOLD    = 50
VOL_SPAN    = 50


def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def _safe_ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])

def _bag_proba(trees, X):
    if not trees: return np.full(len(X), 0.5)
    p = np.zeros(len(X))
    for t in trees:
        cls = list(t.classes_)
        if 1 in cls: p += t.predict_proba(X)[:, cls.index(1)]
    return p / len(trees)

def _fit_naive(X, y, rng):
    n = len(y); trees = []
    for _ in range(N_EST):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2: continue
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                   random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx]); trees.append(t)
    return trees

def _fit_textbook(X, y, t0, t1, n_bars, ms_frac, w_ret, rng):
    n = len(y); m = max(10, int(round(ms_frac * n))); trees = []
    for _ in range(N_EST):
        idx = U.seq_bootstrap(t0, t1, n_bars, m, rng)
        if len(np.unique(y[idx])) < 2: continue
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                   random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx], sample_weight=w_ret[idx]); trees.append(t)
    return trees

def _fit_lossweight(X, y, u_w, rng):
    n = len(y); trees = []
    for _ in range(N_EST):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2: continue
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                   random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx], sample_weight=u_w[idx]); trees.append(t)
    return trees

def load_bars_fx(path):
    try:
        base = tbm.load_base_fx(path)
        thr = base["count"].sum() / N_TARGET
        bars = B.threshold_bars(base, "count", thr)
        return bars if len(bars) >= 3000 else None
    except Exception as e:
        print(f"    load_bars ERROR: {e}"); return None

def make_labels(bars):
    close = bars["close"].to_numpy(np.float64)
    high  = bars["high"].to_numpy(np.float64)
    low   = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    vol = tbm.ewma_vol(close, VOL_SPAN)
    med_abs = np.median(np.abs(log_ret[np.isfinite(log_ret) & (log_ret != 0)]))
    h = max(1e-9, 0.5 * med_abs)
    ev = U.cusum_events(log_ret, h)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]
    if len(ev) < 200: return None
    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    side = side_full[ev].copy(); side[side == 0] = 1
    tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol, PT, SL, MAX_HOLD)
    t0 = ev.astype(np.int64); t1 = np.maximum(tb["touch"].to_numpy(np.int64), t0)
    y = (tb["ret_gross"].to_numpy() > 0).astype(int)
    if y.sum() < 30 or (1-y).sum() < 30: return None
    return dict(t0=t0, t1=t1, y=y, side_full=side_full,
                log_ret=log_ret, n_bars=n, close=close)

def run_cv(bars, lab, name):
    t0, t1, y = lab["t0"], lab["t1"], lab["y"]
    n_bars = lab["n_bars"]; N = len(y)
    feat = tbm.build_features(bars, lab["side_full"],
                              tbm.ewma_vol(lab["close"], VOL_SPAN), FAST, SLOW)
    X = np.nan_to_num(feat.iloc[t0].to_numpy(np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    c_global = U.concurrency(t0, t1, n_bars)
    u_global = U.avg_uniqueness(t0, t1, c_global)
    ms_frac = float(np.clip(u_global.mean(), 0.05, 1.0))
    oos_ll_naive, oos_ll_tb, oos_ll_lw = [], [], []
    is_ll_naive,  is_ll_tb,  is_ll_lw  = [], [], []
    for fold_idx, (tr, te) in enumerate(
            O.purged_kfold_splits(N, N_SPLITS, EMBARGO, LABEL_SPAN)):
        if len(tr) < 80 or y[tr].sum() < 10 or (1-y[tr]).sum() < 10: continue
        lt0 = t0[tr]; lt1 = t1[tr]
        c_tr = U.concurrency(lt0, lt1, n_bars)
        u_tr = U.avg_uniqueness(lt0, lt1, c_tr)
        w_ret_tr = U.return_attribution_weights(lt0, lt1, c_tr, lab["log_ret"])
        if w_ret_tr.sum() > 0: w_ret_tr = w_ret_tr * (len(tr) / w_ret_tr.sum())
        else: w_ret_tr = np.ones(len(tr))
        if u_tr.sum() > 0: u_w = u_tr * (len(tr) / u_tr.sum())
        else: u_w = np.ones(len(tr))
        rng_n  = np.random.default_rng(SEED + fold_idx * 10 + 1)
        rng_tb = np.random.default_rng(SEED + fold_idx * 10 + 2)
        rng_lw = np.random.default_rng(SEED + fold_idx * 10 + 3)
        tn  = _fit_naive(X[tr], y[tr], rng_n)
        ttb = _fit_textbook(X[tr], y[tr], lt0, lt1, n_bars, ms_frac, w_ret_tr, rng_tb)
        tlw = _fit_lossweight(X[tr], y[tr], u_w, rng_lw)
        if not (tn and ttb and tlw): continue
        pn_te  = _bag_proba(tn,  X[te]); ptb_te = _bag_proba(ttb, X[te]); plw_te = _bag_proba(tlw, X[te])
        pn_tr  = _bag_proba(tn,  X[tr]); ptb_tr = _bag_proba(ttb, X[tr]); plw_tr = _bag_proba(tlw, X[tr])
        oos_ll_naive.append(_safe_ll(y[te], pn_te))
        oos_ll_tb.append(   _safe_ll(y[te], ptb_te))
        oos_ll_lw.append(   _safe_ll(y[te], plw_te))
        is_ll_naive.append( _safe_ll(y[tr], pn_tr))
        is_ll_tb.append(    _safe_ll(y[tr], ptb_tr))
        is_ll_lw.append(    _safe_ll(y[tr], plw_tr))
    if not oos_ll_naive: return None
    def _gap(o, i): return float(np.mean(o)) - float(np.mean(i))
    return dict(
        market="forex", name=name, N=N,
        mean_u=float(u_global.mean()), eff_ratio=float(u_global.sum()/N),
        med_oos_naive=float(np.mean(oos_ll_naive)),
        med_oos_tb=   float(np.mean(oos_ll_tb)),
        med_oos_lw=   float(np.mean(oos_ll_lw)),
        gap_naive=_gap(oos_ll_naive, is_ll_naive),
        gap_tb=   _gap(oos_ll_tb,   is_ll_tb),
        gap_lw=   _gap(oos_ll_lw,   is_ll_lw),
        n_folds=len(oos_ll_naive),
        lw_vs_naive_oos=float(np.mean(oos_ll_naive)) - float(np.mean(oos_ll_lw)),
        lw_vs_tb_oos=   float(np.mean(oos_ll_tb))    - float(np.mean(oos_ll_lw)),
        lw_gap_shrink=  _gap(oos_ll_naive, is_ll_naive) - _gap(oos_ll_lw, is_ll_lw),
        # store per-fold for paired tests
        _oos_naive=oos_ll_naive, _oos_tb=oos_ll_tb, _oos_lw=oos_ll_lw,
        _is_naive=is_ll_naive,   _is_tb=is_ll_tb,   _is_lw=is_ll_lw,
    )


def main():
    t_start = time.time()
    new_rows = []

    for sym in FX_SYMS:
        path = os.path.join(FX_DIR, f"{sym}_fx1m.parquet")
        print(f"\n[FX] {sym}  elapsed={( time.time()-t_start)/60:.1f}min  RAM={_peak_mb():.0f}MB", flush=True)
        try:
            bars = load_bars_fx(path)
            if bars is None: print(f"  SKIP: insufficient bars"); continue
            lab = make_labels(bars)
            if lab is None: print(f"  SKIP: insufficient labels"); continue
            N = len(lab["y"])
            print(f"  bars={len(bars)}  labels={N}")
            row = run_cv(bars, lab, sym)
            if row is None: print(f"  SKIP: no valid folds"); continue
            new_rows.append(row)
            print(f"  OOS  naive={row['med_oos_naive']:.4f}  tb={row['med_oos_tb']:.4f}  "
                  f"lw={row['med_oos_lw']:.4f}  gap_naive={row['gap_naive']:+.4f}  gap_lw={row['gap_lw']:+.4f}  folds={row['n_folds']}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}"); traceback.print_exc()
        finally:
            try: del bars, lab
            except NameError: pass
            gc.collect()

    # ── merge with existing parquet ──────────────────────────────────────────
    prev_path = os.path.join(OUTDIR, "s10_per_instrument.parquet")
    df_prev = pd.read_parquet(prev_path)
    print(f"\nPrevious: {len(df_prev)} instruments  New FX: {len(new_rows)}")

    # build FX df (drop per-fold lists before concat)
    if new_rows:
        fx_table = []
        for r in new_rows:
            fx_table.append({k: v for k, v in r.items() if not k.startswith("_")})
        df_fx = pd.DataFrame(fx_table)
        df_all = pd.concat([df_prev, df_fx], ignore_index=True)
    else:
        print("WARNING: No FX rows produced!")
        df_all = df_prev
    df_all.to_parquet(prev_path, index=False)
    print(f"Merged parquet saved: {prev_path}  ({len(df_all)} instruments)")

    # ── aggregate statistics over all instruments ────────────────────────────
    # For paired Wilcoxon: combine prev scalar columns + new FX scalars
    # Pull per-fold sequences from new_rows; for prev rows we only have means
    # (the per-fold lists were not stored in the parquet).
    # Use per-instrument means as the paired test unit (n=42).
    oos_naive = df_all["oos_naive"].to_numpy()
    oos_tb    = df_all["oos_tb"].to_numpy()
    oos_lw    = df_all["oos_lw"].to_numpy()
    gap_naive = df_all["gap_naive"].to_numpy()
    gap_tb    = df_all["gap_tb"].to_numpy()
    gap_lw    = df_all["gap_lw"].to_numpy()
    n = len(df_all)

    wf_lw_vs_naive_oos = float(np.mean(oos_lw <= oos_naive))
    wf_lw_vs_tb_oos    = float(np.mean(oos_lw <= oos_tb))
    wf_lw_gap_vs_naive = float(np.mean(gap_lw <= gap_naive))

    _, p1_two  = ss.wilcoxon(oos_lw - oos_naive, alternative="two-sided")
    _,  p1_less = ss.wilcoxon(oos_lw - oos_naive, alternative="less")
    _,  p2_less = ss.wilcoxon(gap_lw - gap_naive, alternative="less")
    _,  p3_less = ss.wilcoxon(oos_lw - oos_tb,    alternative="less")

    def _band(arr): q25,q50,q75 = np.percentile(arr,[25,50,75]); return q50,q25,q75
    med_on,q25_on,q75_on = _band(oos_naive)
    med_ot,q25_ot,q75_ot = _band(oos_tb)
    med_ol,q25_ol,q75_ol = _band(oos_lw)
    med_gn,q25_gn,q75_gn = _band(gap_naive)
    med_gt,q25_gt,q75_gt = _band(gap_tb)
    med_gl,q25_gl,q75_gl = _band(gap_lw)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {n} instruments")
    print(f"{'='*70}")
    print(f"\nMETRIC SUMMARY (median [Q25,Q75])")
    print(f"  OOS log-loss:  naive={med_on:.4f} [{q25_on:.4f},{q75_on:.4f}]  "
          f"tb={med_ot:.4f} [{q25_ot:.4f},{q75_ot:.4f}]  "
          f"lw={med_ol:.4f} [{q25_ol:.4f},{q75_ol:.4f}]")
    print(f"  Overfit gap:   naive={med_gn:.4f} [{q25_gn:.4f},{q75_gn:.4f}]  "
          f"tb={med_gt:.4f} [{q25_gt:.4f},{q75_gt:.4f}]  "
          f"lw={med_gl:.4f} [{q25_gl:.4f},{q75_gl:.4f}]")
    print(f"\n  Win fractions:")
    print(f"    lw OOS ≤ naive:    {wf_lw_vs_naive_oos:.2%}  ({int(wf_lw_vs_naive_oos*n)}/{n})")
    print(f"    lw OOS ≤ textbook: {wf_lw_vs_tb_oos:.2%}  ({int(wf_lw_vs_tb_oos*n)}/{n})")
    print(f"    lw gap ≤ naive:    {wf_lw_gap_vs_naive:.2%}  ({int(wf_lw_gap_vs_naive*n)}/{n})")
    print(f"\n  Paired Wilcoxon:")
    print(f"    [1] lw OOS ≤ naive    p_less={p1_less:.4f}  p_two={p1_two:.4f}")
    print(f"    [2] lw gap < naive    p_less={p2_less:.4f}")
    print(f"    [3] lw OOS < textbook p_less={p3_less:.4f}")

    ALPHA = 0.05
    bar1 = bool((wf_lw_vs_naive_oos >= 0.50) and (p1_less < ALPHA))
    bar2 = bool((wf_lw_gap_vs_naive >= 0.50) and (p2_less < ALPHA))
    bar3 = bool((wf_lw_vs_tb_oos  >= 0.50) and (p3_less < ALPHA))
    verdict = "PASS" if (bar1 and bar2 and bar3) else "FAIL"
    print(f"\n  Bar check:  [1]={'MET' if bar1 else 'MISSED'}  [2]={'MET' if bar2 else 'MISSED'}  [3]={'MET' if bar3 else 'MISSED'}")
    print(f"  VERDICT: {verdict}")

    summary = dict(
        verdict=verdict, n_instruments=int(n),
        med_oos_naive=float(med_on),  med_oos_naive_q25=float(q25_on),  med_oos_naive_q75=float(q75_on),
        med_oos_tb=float(med_ot),     med_oos_tb_q25=float(q25_ot),     med_oos_tb_q75=float(q75_ot),
        med_oos_lw=float(med_ol),     med_oos_lw_q25=float(q25_ol),     med_oos_lw_q75=float(q75_ol),
        med_gap_naive=float(med_gn),  med_gap_naive_q25=float(q25_gn),  med_gap_naive_q75=float(q75_gn),
        med_gap_tb=float(med_gt),     med_gap_tb_q25=float(q25_gt),     med_gap_tb_q75=float(q75_gt),
        med_gap_lw=float(med_gl),     med_gap_lw_q25=float(q25_gl),     med_gap_lw_q75=float(q75_gl),
        wf_lw_vs_naive_oos=float(wf_lw_vs_naive_oos),
        wf_lw_vs_tb_oos=float(wf_lw_vs_tb_oos),
        wf_lw_gap_vs_naive=float(wf_lw_gap_vs_naive),
        p_lw_oos_le_naive_onesided=float(p1_less),
        p_lw_oos_le_naive_twosided=float(p1_two),
        p_lw_gap_lt_naive=float(p2_less),
        p_lw_oos_lt_tb=float(p3_less),
        bar1_met=bar1, bar2_met=bar2, bar3_met=bar3,
        run_time_min=float((time.time()-t_start)/60),
        peak_ram_mb=float(_peak_mb()),
    )
    sum_path = os.path.join(OUTDIR, "s10_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {sum_path}")
    print(f"  Peak RAM: {_peak_mb():.0f}MB")
    print("\nDONE.")


if __name__ == "__main__":
    main()

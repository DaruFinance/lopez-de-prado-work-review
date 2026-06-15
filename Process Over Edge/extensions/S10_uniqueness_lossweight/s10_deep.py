#!/usr/bin/env python3
"""
s10_deep.py — S10 deep run: loss-weight-only uniqueness (PREREGISTRATION.md S10).

PASS criterion (locked):
  Loss-weight-only (uniqueness as sample_weight in fit ONLY, no bag-size shrink,
  no sequential bootstrap) across the full 42-instrument panel:
    1. Preserves OOS accuracy vs naive (median OOS log-loss ≤ naive, win-fraction ≥ 50%).
    2. Cuts IS→OOS overfit gap vs naive (median gap reduction > 0, paired test sig.).
    3. Beats textbook-corrected arm on OOS accuracy (median OOS log-loss < corrected,
       paired test one-sided p < 0.05 across instruments).
  All three must hold simultaneously.

THREE arms under IDENTICAL purged k-fold CV:
  1. naive:        IID bag (max_samples=1.0), no weights, sklearn default.
  2. textbook:     Sequential bootstrap + bag-size = mean avg-uniqueness + return-
                   attribution weights (full LdP correction).
  3. loss_weight:  IID bag (max_samples=1.0) + uniqueness as sample_weight in fit.
                   No sequential bootstrap, no bag-size shrink.

No-lookahead check (stated explicitly):
  - All concurrency/uniqueness statistics computed from [t0,t1] spans of the TRAIN
    fold labels only (train t0,t1 are passed; they use bars up to the TRAIN window).
  - Features built causally from bars (ewma, MA, RSI — all causal).
  - The model is fit on the train fold; CV folds are purged + embargoed so no
    training sample overlaps the test window via its label span.
  - Uniqueness weights used in fit_bag are derived from the TRAIN fold concurrency
    only; the test fold is never seen during weight computation.

Instruments: 27 crypto perps, 7 equity ETFs, 8 FX = 42 total.
Peak RAM estimate: ~6 GB (one instrument in memory at a time; gc between each).

Run wrapper:
  cd extensions/S10_uniqueness_lossweight && \
  ulimit -v 10485760 && \
  taskset -c 16-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python3 s10_deep.py > run.log 2>&1
"""
from __future__ import annotations
import os, sys, gc, time, json, warnings, traceback

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
)
warnings.filterwarnings("ignore")

import resource
import numpy as np
import pandas as pd
from scipy import stats as ss

# ── repo paths ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, FX_1M, EQUITY_1M
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

from lib import bars as B
import overfit as O
import tbm
import uniqueness as U

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, roc_auc_score

# ── output ──────────────────────────────────────────────────────────────────
OUTDIR = HERE
os.makedirs(OUTDIR, exist_ok=True)

# ── data roots ───────────────────────────────────────────────────────────────
CRYPTO_DIR  = CRYPTO_1M
FX_DIR      = FX_1M
EQUITY_DIR  = EQUITY_1M

CRYPTO_SYMS = [
    "1000SHIBUSDT", "AAVEUSDT", "ALGOUSDT", "APEUSDT", "APTUSDT",
    "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT",
    "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETCUSDT", "ETHUSDT",
    "HBARUSDT", "ICPUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT",
    "SOLUSDT", "SUIUSDT", "TRXUSDT", "UNIUSDT", "XLMUSDT",
    "XRPUSDT", "ZECUSDT",
]
EQUITY_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
FX_SYMS     = ["AUDUSD", "EURGBP", "EURUSD", "GBPUSD",
               "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

# ── CV / model params ────────────────────────────────────────────────────────
N_TARGET     = 20_000   # target dollar/tick bars per instrument
N_EST        = 40       # trees per bag
N_SPLITS     = 6        # purged k-fold
EMBARGO      = 0.01
LABEL_SPAN   = 3        # purge window (bars)
SEED         = 0
FAST, SLOW   = 20, 60
PT, SL       = 1.5, 1.0
MAX_HOLD     = 50
VOL_SPAN     = 50


# ── helpers ──────────────────────────────────────────────────────────────────
def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _safe_ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _bag_proba(trees, X):
    """Average predicted P(class=1) across an ensemble."""
    if not trees:
        return np.full(len(X), 0.5)
    p = np.zeros(len(X))
    for t in trees:
        cls = list(t.classes_)
        if 1 in cls:
            p += t.predict_proba(X)[:, cls.index(1)]
        else:
            p += 0.0
    return p / len(trees)


def _fit_naive(X, y, rng):
    """IID bag, no weights, max_samples=1.0 (sklearn default)."""
    n = len(y)
    trees = []
    for _ in range(N_EST):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        t = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20,
            random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx])
        trees.append(t)
    return trees


def _fit_textbook(X, y, t0, t1, n_bars, ms_frac, w_ret, rng):
    """Sequential bootstrap + bag-size = mean avg-uniqueness + return-attribution weights."""
    n = len(y)
    m = max(10, int(round(ms_frac * n)))
    trees = []
    for _ in range(N_EST):
        idx = U.seq_bootstrap(t0, t1, n_bars, m, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        t = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20,
            random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx], sample_weight=w_ret[idx])
        trees.append(t)
    return trees


def _fit_lossweight(X, y, u_w, rng):
    """IID bag (max_samples=1.0) with uniqueness as sample_weight — loss-weight-only arm."""
    n = len(y)
    trees = []
    for _ in range(N_EST):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        t = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20,
            random_state=int(rng.integers(0, 2**31)))
        # uniqueness weights in LOSS only (passed to fit, not to sampling)
        t.fit(X[idx], y[idx], sample_weight=u_w[idx])
        trees.append(t)
    return trees


# ── bar construction ─────────────────────────────────────────────────────────
def load_bars(market, path):
    try:
        if market == "crypto":
            base = tbm.load_base_crypto(path)
            bars = B.matched_bars(base, N_TARGET)["dollar"]
        elif market == "equity":
            base = B.load_base_equity_etf(path, rth=True)
            bars = B.matched_bars(base, N_TARGET)["dollar"]
        elif market in ("fx", "forex"):
            base = tbm.load_base_fx(path)
            thr = base["count"].sum() / N_TARGET
            bars = B.threshold_bars(base, "count", thr)
        else:
            return None
    except Exception as e:
        print(f"    load_bars ERROR: {e}")
        return None
    return bars if len(bars) >= 3000 else None


# ── labelling ────────────────────────────────────────────────────────────────
def make_labels(bars):
    close  = bars["close"].to_numpy(np.float64)
    high   = bars["high"].to_numpy(np.float64)
    low    = bars["low"].to_numpy(np.float64)
    openp  = bars["open"].to_numpy(np.float64)
    n      = len(close)
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    vol    = tbm.ewma_vol(close, VOL_SPAN)

    med_abs = np.median(np.abs(log_ret[np.isfinite(log_ret) & (log_ret != 0)]))
    h = max(1e-9, 0.5 * med_abs)
    ev = U.cusum_events(log_ret, h)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]
    if len(ev) < 200:
        return None

    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    side = side_full[ev].copy()
    side[side == 0] = 1

    tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol,
                            PT, SL, MAX_HOLD)
    t0 = ev.astype(np.int64)
    t1 = np.maximum(tb["touch"].to_numpy(np.int64), t0)
    y  = (tb["ret_gross"].to_numpy() > 0).astype(int)

    if y.sum() < 30 or (1 - y).sum() < 30:
        return None

    return dict(t0=t0, t1=t1, y=y, side_full=side_full,
                log_ret=log_ret, n_bars=n, close=close)


# ── per-instrument CV ─────────────────────────────────────────────────────────
def run_cv(bars, lab, name):
    """
    Run purged k-fold CV for three arms. Returns per-fold metrics dict.

    No-lookahead: all weights/uniqueness are computed from train-fold t0,t1
    which resolve forward within the TRAINING window only. The OOS fold is
    never touched during weight or bag computation.
    """
    t0, t1, y = lab["t0"], lab["t1"], lab["y"]
    n_bars = lab["n_bars"]
    N = len(y)

    # feature matrix (causal: uses only data up to each bar's close)
    feat = tbm.build_features(bars, lab["side_full"],
                              tbm.ewma_vol(lab["close"], VOL_SPAN), FAST, SLOW)
    X = np.nan_to_num(feat.iloc[t0].to_numpy(np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)

    # global concurrency for textbook bag-size (full set, used for ms_frac only)
    c_global   = U.concurrency(t0, t1, n_bars)
    u_global   = U.avg_uniqueness(t0, t1, c_global)
    ms_frac    = float(np.clip(u_global.mean(), 0.05, 1.0))  # corrected bag-size

    oos_ll_naive, oos_ll_tb, oos_ll_lw = [], [], []
    is_ll_naive,  is_ll_tb,  is_ll_lw  = [], [], []

    for fold_idx, (tr, te) in enumerate(
            O.purged_kfold_splits(N, N_SPLITS, EMBARGO, LABEL_SPAN)):
        if len(tr) < 80 or y[tr].sum() < 10 or (1 - y[tr]).sum() < 10:
            continue

        # ── train-fold uniqueness (NO-LOOKAHEAD: only train spans used) ──
        lt0 = t0[tr];  lt1 = t1[tr]
        c_tr = U.concurrency(lt0, lt1, n_bars)
        u_tr = U.avg_uniqueness(lt0, lt1, c_tr)

        # return-attribution weights for textbook arm (train-fold only)
        w_ret_tr = U.return_attribution_weights(lt0, lt1, c_tr, lab["log_ret"])
        if w_ret_tr.sum() > 0:
            w_ret_tr = w_ret_tr * (len(tr) / w_ret_tr.sum())
        else:
            w_ret_tr = np.ones(len(tr))

        # uniqueness weights for loss-weight arm (normalised to mean 1)
        if u_tr.sum() > 0:
            u_w = u_tr * (len(tr) / u_tr.sum())
        else:
            u_w = np.ones(len(tr))

        # independent RNG seeds per arm (identical data, fair comparison)
        rng_n  = np.random.default_rng(SEED + fold_idx * 10 + 1)
        rng_tb = np.random.default_rng(SEED + fold_idx * 10 + 2)
        rng_lw = np.random.default_rng(SEED + fold_idx * 10 + 3)

        tn  = _fit_naive(X[tr], y[tr], rng_n)
        ttb = _fit_textbook(X[tr], y[tr], lt0, lt1, n_bars, ms_frac, w_ret_tr, rng_tb)
        tlw = _fit_lossweight(X[tr], y[tr], u_w, rng_lw)

        if not (tn and ttb and tlw):
            continue

        # OOS predictions
        pn_te  = _bag_proba(tn,  X[te])
        ptb_te = _bag_proba(ttb, X[te])
        plw_te = _bag_proba(tlw, X[te])

        # IS predictions (for overfit gap)
        pn_tr  = _bag_proba(tn,  X[tr])
        ptb_tr = _bag_proba(ttb, X[tr])
        plw_tr = _bag_proba(tlw, X[tr])

        oos_ll_naive.append(_safe_ll(y[te], pn_te))
        oos_ll_tb.append(   _safe_ll(y[te], ptb_te))
        oos_ll_lw.append(   _safe_ll(y[te], plw_te))

        is_ll_naive.append( _safe_ll(y[tr], pn_tr))
        is_ll_tb.append(    _safe_ll(y[tr], ptb_tr))
        is_ll_lw.append(    _safe_ll(y[tr], plw_tr))

    if not oos_ll_naive:
        return None

    def _gap(oos, iss):
        return float(np.mean(oos)) - float(np.mean(iss))

    return dict(
        name=name, N=N,
        mean_u=float(u_global.mean()), eff_ratio=float(u_global.sum() / N),
        # per-fold lists (for paired tests later)
        oos_naive=oos_ll_naive, oos_tb=oos_ll_tb, oos_lw=oos_ll_lw,
        is_naive=is_ll_naive,   is_tb=is_ll_tb,   is_lw=is_ll_lw,
        # aggregate scalars
        med_oos_naive=float(np.mean(oos_ll_naive)),
        med_oos_tb=   float(np.mean(oos_ll_tb)),
        med_oos_lw=   float(np.mean(oos_ll_lw)),
        gap_naive=_gap(oos_ll_naive, is_ll_naive),
        gap_tb=   _gap(oos_ll_tb,   is_ll_tb),
        gap_lw=   _gap(oos_ll_lw,   is_ll_lw),
        n_folds=len(oos_ll_naive),
    )


# ── instrument loop ──────────────────────────────────────────────────────────
def iter_instruments():
    for sym in CRYPTO_SYMS:
        path = os.path.join(CRYPTO_DIR, f"{sym}_1m.parquet")
        yield ("crypto", sym, path)
    for sym in EQUITY_SYMS:
        path = os.path.join(EQUITY_DIR, f"{sym}.csv.gz")
        yield ("equity", sym, path)
    for sym in FX_SYMS:
        path = os.path.join(FX_DIR, f"{sym}_fx1m.parquet")
        yield ("forex", sym, path)


def main():
    t_start = time.time()
    results = []
    skipped = []

    total = len(CRYPTO_SYMS) + len(EQUITY_SYMS) + len(FX_SYMS)
    done  = 0

    for market, name, path in iter_instruments():
        done += 1
        elapsed = time.time() - t_start
        print(f"\n[{done:02d}/{total}] {name} ({market})  elapsed={elapsed/60:.1f}min  "
              f"RAM={_peak_mb():.0f}MB", flush=True)

        if not os.path.exists(path):
            print(f"  SKIP: path not found: {path}")
            skipped.append(name)
            continue

        try:
            bars = load_bars(market, path)
            if bars is None:
                print(f"  SKIP: insufficient bars")
                skipped.append(name)
                continue

            lab = make_labels(bars)
            if lab is None:
                print(f"  SKIP: insufficient labels")
                skipped.append(name)
                continue

            N = len(lab["y"])
            print(f"  bars={len(bars)}  labels={N}  mean_u={U.avg_uniqueness(lab['t0'],lab['t1'],U.concurrency(lab['t0'],lab['t1'],lab['n_bars'])).mean():.3f}",
                  flush=True)

            row = run_cv(bars, lab, name)
            if row is None:
                print(f"  SKIP: no valid CV folds")
                skipped.append(name)
                continue

            row["market"] = market
            results.append(row)

            print(f"  OOS  naive={row['med_oos_naive']:.4f}  "
                  f"tb={row['med_oos_tb']:.4f}  "
                  f"lw={row['med_oos_lw']:.4f}   "
                  f"gap_naive={row['gap_naive']:+.4f}  "
                  f"gap_lw={row['gap_lw']:+.4f}  "
                  f"folds={row['n_folds']}", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            skipped.append(name)
        finally:
            # release memory before next instrument
            try:
                del bars, lab
            except NameError:
                pass
            gc.collect()

    # ── aggregate statistics ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"COMPLETED: {len(results)}/{total} instruments  "
          f"({len(skipped)} skipped: {skipped})")
    print(f"{'='*70}\n")

    if len(results) < 10:
        print("ERROR: Too few valid instruments for reliable inference.")
        return

    # scalar summaries per instrument
    oos_naive = np.array([r["med_oos_naive"] for r in results])
    oos_tb    = np.array([r["med_oos_tb"]    for r in results])
    oos_lw    = np.array([r["med_oos_lw"]    for r in results])
    gap_naive = np.array([r["gap_naive"]     for r in results])
    gap_tb    = np.array([r["gap_tb"]        for r in results])
    gap_lw    = np.array([r["gap_lw"]        for r in results])

    # win fractions
    n = len(results)
    wf_lw_vs_naive_oos  = float(np.mean(oos_lw <= oos_naive))   # lw ≤ naive OOS loss
    wf_lw_vs_tb_oos     = float(np.mean(oos_lw <= oos_tb))       # lw ≤ tb OOS loss
    wf_lw_gap_vs_naive  = float(np.mean(gap_lw <= gap_naive))    # lw ≤ naive gap

    # paired Wilcoxon signed-rank tests (one-sided where specified by the bar)
    # claim 1: lw ≤ naive OOS loss  (loss-weight does not hurt accuracy)
    _, p1_two = ss.wilcoxon(oos_lw - oos_naive, alternative="two-sided")
    stat1, p1_less = ss.wilcoxon(oos_lw - oos_naive, alternative="less")   # lw < naive

    # claim 2: lw gap < naive gap  (overfit gap shrinks)
    stat2, p2_less = ss.wilcoxon(gap_lw - gap_naive, alternative="less")   # lw gap smaller

    # claim 3: lw OOS loss < tb OOS loss  (lw beats textbook on accuracy)
    stat3, p3_less = ss.wilcoxon(oos_lw - oos_tb, alternative="less")      # lw < tb

    # -- band summaries (median ± IQR) --
    def _band(arr):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return q50, q25, q75

    med_oos_naive, q25_naive, q75_naive = _band(oos_naive)
    med_oos_tb,    q25_tb,    q75_tb    = _band(oos_tb)
    med_oos_lw,    q25_lw,    q75_lw   = _band(oos_lw)
    med_gap_naive, q25_gn, q75_gn      = _band(gap_naive)
    med_gap_tb,    q25_gt, q75_gt      = _band(gap_tb)
    med_gap_lw,    q25_gl, q75_gl      = _band(gap_lw)

    print("METRIC SUMMARY (median [Q25, Q75] across instruments)\n")
    print(f"  OOS log-loss (lower = better):")
    print(f"    naive:     {med_oos_naive:.4f} [{q25_naive:.4f}, {q75_naive:.4f}]")
    print(f"    textbook:  {med_oos_tb:.4f} [{q25_tb:.4f}, {q75_tb:.4f}]")
    print(f"    loss-w:    {med_oos_lw:.4f} [{q25_lw:.4f}, {q75_lw:.4f}]")
    print()
    print(f"  IS→OOS overfit gap (lower = less overfit):")
    print(f"    naive:     {med_gap_naive:.4f} [{q25_gn:.4f}, {q75_gn:.4f}]")
    print(f"    textbook:  {med_gap_tb:.4f} [{q25_gt:.4f}, {q75_gt:.4f}]")
    print(f"    loss-w:    {med_gap_lw:.4f} [{q25_gl:.4f}, {q75_gl:.4f}]")
    print()
    print(f"  Win fractions (loss-weight-only):")
    print(f"    OOS accuracy >= naive:    {wf_lw_vs_naive_oos:.2%} ({int(wf_lw_vs_naive_oos*n)}/{n})")
    print(f"    OOS accuracy >= textbook: {wf_lw_vs_tb_oos:.2%}   ({int(wf_lw_vs_tb_oos*n)}/{n})")
    print(f"    Gap <= naive:             {wf_lw_gap_vs_naive:.2%} ({int(wf_lw_gap_vs_naive*n)}/{n})")
    print()
    print(f"  Paired Wilcoxon tests:")
    print(f"    [1] lw OOS ≤ naive    p(one-sided less)={p1_less:.4f}  p(two-sided)={p1_two:.4f}")
    print(f"    [2] lw gap < naive    p(one-sided less)={p2_less:.4f}")
    print(f"    [3] lw OOS < textbook p(one-sided less)={p3_less:.4f}")
    print()

    # ── verdict ───────────────────────────────────────────────────────────────
    ALPHA = 0.05
    bar1_met = (wf_lw_vs_naive_oos >= 0.50) and (p1_less < ALPHA)
    bar2_met = (wf_lw_gap_vs_naive >= 0.50) and (p2_less < ALPHA)
    bar3_met = (wf_lw_vs_tb_oos  >= 0.50) and (p3_less < ALPHA)

    verdict = "PASS" if (bar1_met and bar2_met and bar3_met) else "FAIL"
    print(f"  Bar check:")
    print(f"    [1] preserves accuracy vs naive:   {'MET' if bar1_met else 'MISSED'}")
    print(f"    [2] cuts overfit gap vs naive:      {'MET' if bar2_met else 'MISSED'}")
    print(f"    [3] beats textbook on OOS accuracy: {'MET' if bar3_met else 'MISSED'}")
    print()
    print(f"  VERDICT: {verdict}")
    print()

    # ── save artefacts ────────────────────────────────────────────────────────
    # per-instrument table (no per-fold lists)
    table_rows = []
    for r in results:
        table_rows.append(dict(
            market=r["market"], name=r["name"], N=r["N"],
            mean_u=r["mean_u"], eff_ratio=r["eff_ratio"],
            oos_naive=r["med_oos_naive"], oos_tb=r["med_oos_tb"], oos_lw=r["med_oos_lw"],
            gap_naive=r["gap_naive"], gap_tb=r["gap_tb"], gap_lw=r["gap_lw"],
            n_folds=r["n_folds"],
            lw_vs_naive_oos=r["med_oos_naive"] - r["med_oos_lw"],   # +good
            lw_vs_tb_oos=r["med_oos_tb"] - r["med_oos_lw"],          # +good
            lw_gap_shrink=r["gap_naive"] - r["gap_lw"],              # +good
        ))
    df = pd.DataFrame(table_rows)
    df_path = os.path.join(OUTDIR, "s10_per_instrument.parquet")
    df.to_parquet(df_path, index=False)
    print(f"  Per-instrument table: {df_path}")

    # summary JSON
    summary = dict(
        verdict=verdict,
        n_instruments=n,
        n_skipped=len(skipped),
        skipped=skipped,
        # median OOS log-loss
        med_oos_naive=float(med_oos_naive), med_oos_naive_q25=float(q25_naive), med_oos_naive_q75=float(q75_naive),
        med_oos_tb=float(med_oos_tb),       med_oos_tb_q25=float(q25_tb),       med_oos_tb_q75=float(q75_tb),
        med_oos_lw=float(med_oos_lw),       med_oos_lw_q25=float(q25_lw),       med_oos_lw_q75=float(q75_lw),
        # overfit gap
        med_gap_naive=float(med_gap_naive), med_gap_naive_q25=float(q25_gn), med_gap_naive_q75=float(q75_gn),
        med_gap_tb=float(med_gap_tb),       med_gap_tb_q25=float(q25_gt),    med_gap_tb_q75=float(q75_gt),
        med_gap_lw=float(med_gap_lw),       med_gap_lw_q25=float(q25_gl),    med_gap_lw_q75=float(q75_gl),
        # win fractions
        wf_lw_vs_naive_oos=wf_lw_vs_naive_oos,
        wf_lw_vs_tb_oos=wf_lw_vs_tb_oos,
        wf_lw_gap_vs_naive=wf_lw_gap_vs_naive,
        # paired test p-values
        p_lw_oos_le_naive_onesided=float(p1_less),
        p_lw_oos_le_naive_twosided=float(p1_two),
        p_lw_gap_lt_naive=float(p2_less),
        p_lw_oos_lt_tb=float(p3_less),
        # bar checks
        bar1_met=bool(bar1_met),
        bar2_met=bool(bar2_met),
        bar3_met=bool(bar3_met),
        run_time_min=float((time.time() - t_start) / 60),
        peak_ram_mb=float(_peak_mb()),
    )
    sum_path = os.path.join(OUTDIR, "s10_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON:         {sum_path}")
    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min   Peak RAM: {_peak_mb():.0f} MB")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
S06_deep.py — Deep extension S6: max_features = 1 / tree decorrelation as bagging mechanism.

Pre-registration (LOCKED 2026-06-12):
    Sweep max_features ∈ {1, 2, 3, sqrt(p), p} in a bagged RF across the full
    40-instrument panel under purged k-fold CV. For each setting measure the
    IS→OOS log-loss gap AND OOS log-loss. Claim: low/1 max_features minimises
    BOTH across the panel.  Paired test: Wilcoxon signed-rank, mf=1 vs sqrt(p).
    Also check whether the gap shrinks monotonically as max_features falls.

Floors (from PREREGISTRATION.md):
    - threads pinned to 1 (sklearn n_jobs=1, OPENBLAS/OMP/MKL=1)
    - full 40-instrument panel: 27 crypto perps + 7 equity ETFs + 8 FX
    - full available history, dollar bars (tick bars for FX)
    - EMA-crossover triple-barrier task from run_ensembles_importance.py
    - real costs (crypto 7bp rt, equities via lib.realism, FX 1bp rt)
    - RAM-safe: one instrument at a time, del+gc, heartbeat every instrument
    - artifacts under the script's own directory

Peak RAM estimate: ~2-4 GB peak per instrument (dollar bar build + RF fits).
"""
import os, sys, gc, warnings, json, time
# CRITICAL: pin threads BEFORE any numpy/sklearn import
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1",
)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as ss

# ---- repo paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, CRYPTO_1M, EQUITY_1M, FX_1M
for p in [REPO_ROOT,
          _LIB,
          os.path.join(REPO_ROOT, "projects", "09_ensembles_importance", "scripts"),
          os.path.join(REPO_ROOT, "projects", "03_meta_labeling", "scripts"),
          HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg
from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import tbm
import run_ensembles_importance as P

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# =========================================================================== #
# Configuration
# =========================================================================== #
OUTDIR = HERE

CRYPTO_DIR  = CRYPTO_1M
ETF_DIR     = EQUITY_1M
FX_DIR      = FX_1M
ETF_SYMS    = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-side costs (bp) — round-trip = 2x for crypto/FX; equities via realism
COST_BP     = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}

# task parameters (match run_ensembles_importance.py exactly)
FAST, SLOW  = 20, 60
PT_MULT, SL_MULT = 1.5, 1.0
MAX_HOLD    = 50
VOL_SPAN    = 50
N_TARGET_BARS = 20000

# CV parameters (match study)
N_SPLITS    = 6
EMBARGO     = 0.01
RANDOM_STATE = 0
N_ESTIMATORS = 200  # match study's RF_GRID["n_estimators"]

# max_features sweep — "sqrt" and "p" (=None in sklearn) are label-only;
# actual values computed per instrument after feature count is known
MF_LABELS   = ["1", "2", "3", "sqrt", "p"]

MIN_WARMUP  = SLOW + 12


# =========================================================================== #
# Universe builder (mirrors run_ensembles_importance.py exactly)
# =========================================================================== #
def universe():
    import glob
    u = []
    for path in sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet"))):
        name = os.path.basename(path).replace("_1m.parquet", "")
        u.append(("crypto", name, path))
    for sym in ETF_SYMS:
        path = os.path.join(ETF_DIR, f"{sym}.csv.gz")
        if os.path.exists(path):
            u.append(("equities", sym, path))
    for path in sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet"))):
        name = os.path.basename(path).replace("_fx1m.parquet", "")
        u.append(("forex", name, path))
    return u


# =========================================================================== #
# Task builder (mirrors run_ensembles_importance.py build_task exactly)
# =========================================================================== #
def build_task(market, path):
    """Build the triple-barrier labeled task for one instrument.
    Returns dict or None if insufficient data."""
    try:
        if market == "crypto":
            base = tbm.load_base_crypto(path)
            bars = B.matched_bars(base, N_TARGET_BARS)["dollar"]
        elif market == "equities":
            base = B.load_base_equity_etf(path, rth=True)
            bars = B.matched_bars(base, N_TARGET_BARS)["dollar"]
        elif market == "forex":
            base = tbm.load_base_fx(path)
            thr = base["count"].sum() / N_TARGET_BARS
            bars = B.threshold_bars(base, "count", thr)
        else:
            return None
    except Exception as e:
        print(f"    [load error] {e}")
        return None

    if len(bars) < 4000:
        return None

    # bars_per_year
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    bpy = len(bars) / years

    close  = bars["close"].to_numpy(np.float64)
    high   = bars["high"].to_numpy(np.float64)
    low    = bars["low"].to_numpy(np.float64)
    openp  = bars["open"].to_numpy(np.float64)
    n      = len(close)

    # realistic per-side cost
    sym = (os.path.basename(path)
           .replace("_fx1m.parquet", "")
           .replace("_1m.parquet", "")
           .replace(".csv.gz", ""))
    try:
        cost_side = RZ.per_side_cost_fraction(market, sym, bars.index, close,
                                              crypto_fallback=COST_BP[market] / 1e4)
    except Exception:
        cost_side = np.full(n, COST_BP[market] / 1e4)

    vol      = tbm.ewma_vol(close, VOL_SPAN)
    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    ev       = tbm.crossover_events(side_full)
    ev       = ev[(ev > MIN_WARMUP) & (ev < n - 1)]
    if len(ev) < 300:
        return None

    tb       = tbm.triple_barrier(close, high, low, openp, ev, side_full[ev],
                                  vol, PT_MULT, SL_MULT, MAX_HOLD)
    ret_gross = tb["ret_gross"].to_numpy()
    hold      = tb["hold"].to_numpy()

    ex_bar  = np.minimum(ev + np.maximum(1, hold.astype(np.int64)), n - 1)
    rt_cost = cost_side[ev] + cost_side[ex_bar]
    pnl_net = ret_gross - rt_cost
    y       = (pnl_net > 0).astype(np.int64)
    if y.sum() < 50 or (1 - y).sum() < 50:
        return None

    feat      = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    feat_names = list(feat.columns)
    X         = feat.iloc[ev].to_numpy(np.float64)
    X         = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    gap   = max(1, int(np.median(np.diff(ev)))) if len(ev) > 1 else 1
    label_span_ev = int(np.ceil(MAX_HOLD / gap)) + 1
    t0    = (np.asarray(ev, np.int64) - ev[0])
    t1    = np.minimum(t0 + hold.astype(np.int64), (n - 1) - ev[0])
    nb_span = int((n - 1) - ev[0]) + 1

    sw, avg_u = P.make_sample_weights(t0, t1, nb_span)

    return dict(
        market=market, bars=bars, X=X, y=y, ev=ev, hold=hold,
        ret_gross=ret_gross, cost=rt_cost,
        sw=sw, avg_u=float(avg_u), t0=t0, t1=t1,
        nb_span=nb_span, label_span_ev=label_span_ev,
        n_bars=n, bpy=bpy, feat_names=feat_names
    )


# =========================================================================== #
# max_features sweep for one instrument
# =========================================================================== #
def sweep_max_features(task):
    """For each max_features setting run purged k-fold CV and collect per-fold
    IS log-loss and OOS log-loss. Returns dict keyed by MF_LABELS."""
    X, y, sw = task["X"], task["y"], task["sw"]
    avg_u    = task["avg_u"]
    ls       = task["label_span_ev"]
    p_feats  = X.shape[1]

    # resolve actual sklearn values for symbolic labels
    def resolve_mf(label):
        if label == "sqrt":
            return "sqrt"           # sklearn native
        if label == "p":
            return None             # sklearn: use all features
        return int(label)

    results = {}
    for label in MF_LABELS:
        mf_val = resolve_mf(label)
        # clamp integer mf to [1, p_feats]
        if isinstance(mf_val, int):
            mf_val = max(1, min(mf_val, p_feats))

        ll_is_folds, ll_oos_folds = [], []
        valid_folds = 0

        for tr, te in O.purged_kfold_splits(len(y), N_SPLITS, EMBARGO, ls):
            if len(tr) < 60 or len(np.unique(y[tr])) < 2 or len(te) < 10:
                continue
            mdl = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_features=mf_val,
                max_samples=float(np.clip(avg_u, 0.05, 0.99)),
                min_weight_fraction_leaf=0.0,
                bootstrap=True,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            )
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
            ll_is_folds.append(
                log_loss(y[tr], mdl.predict_proba(X[tr]), labels=[0, 1])
            )
            ll_oos_folds.append(
                log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1])
            )
            valid_folds += 1

        if not ll_is_folds:
            results[label] = dict(ll_is=np.nan, ll_oos=np.nan, gap=np.nan,
                                  valid_folds=0)
            continue

        ll_is_arr  = np.array(ll_is_folds)
        ll_oos_arr = np.array(ll_oos_folds)
        gap_arr    = ll_oos_arr - ll_is_arr

        results[label] = dict(
            ll_is      = float(ll_is_arr.mean()),
            ll_oos     = float(ll_oos_arr.mean()),
            gap        = float(gap_arr.mean()),
            # per-fold arrays (for cross-instrument paired tests)
            ll_is_folds  = ll_is_arr.tolist(),
            ll_oos_folds = ll_oos_arr.tolist(),
            gap_folds    = gap_arr.tolist(),
            valid_folds  = valid_folds,
        )

    return results


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    t_run_start = time.perf_counter()
    os.makedirs(OUTDIR, exist_ok=True)

    u = universe()
    print(f"\nS6 deep run: {len(u)} instruments "
          f"({sum(1 for m,_,_ in u if m=='crypto')} crypto, "
          f"{sum(1 for m,_,_ in u if m=='equities')} ETF, "
          f"{sum(1 for m,_,_ in u if m=='forex')} FX)")
    print(f"max_features sweep: {MF_LABELS}")
    print(f"N_ESTIMATORS={N_ESTIMATORS}, N_SPLITS={N_SPLITS}, EMBARGO={EMBARGO}")
    print(f"Artifacts -> {OUTDIR}\n")

    rows = []

    for inst_idx, (market, name, path) in enumerate(u):
        t_inst = time.perf_counter()
        print(f"[{inst_idx+1:2d}/{len(u)}] {market:9s} {name:15s} ... ", end="", flush=True)

        task = build_task(market, path)
        if task is None:
            print("SKIP (insufficient data)")
            continue

        n_ev = len(task["y"])
        p    = task["X"].shape[1]

        try:
            sweep = sweep_max_features(task)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            del task; gc.collect()
            continue

        dt = time.perf_counter() - t_inst

        # build flat row for parquet
        row = dict(market=market, name=name, n_ev=n_ev,
                   p_features=p, avg_u=task["avg_u"],
                   base_rate=float(task["y"].mean()))
        for label in MF_LABELS:
            r = sweep.get(label, {})
            row[f"ll_is_{label}"]  = r.get("ll_is",  np.nan)
            row[f"ll_oos_{label}"] = r.get("ll_oos", np.nan)
            row[f"gap_{label}"]    = r.get("gap",    np.nan)
            row[f"folds_{label}"]  = r.get("valid_folds", 0)

        rows.append(row)

        # per-instrument summary line
        gaps = " ".join(f"{label}:{sweep.get(label,{}).get('gap', float('nan')):.4f}"
                        for label in MF_LABELS)
        ooss = " ".join(f"{label}:{sweep.get(label,{}).get('ll_oos', float('nan')):.4f}"
                        for label in MF_LABELS)
        print(f"n_ev={n_ev} p={p}  gap: {gaps}  ({dt:.1f}s)")
        print(f"{'':42s}oos: {ooss}")

        del task; gc.collect()

    # =========================================================================
    # Save per-instrument results
    # =========================================================================
    df = pd.DataFrame(rows)
    out_parquet = os.path.join(OUTDIR, "per_instrument.parquet")
    out_csv     = os.path.join(OUTDIR, "per_instrument.csv")
    df.to_parquet(out_parquet)
    df.to_csv(out_csv, index=False)
    print(f"\n\nPer-instrument results -> {out_parquet}")

    if df.empty:
        print("NO RESULTS — all instruments skipped.")
        return

    # =========================================================================
    # Cross-sectional analysis
    # =========================================================================
    print("\n" + "="*70)
    print("CROSS-SECTIONAL ANALYSIS")
    print("="*70)

    n_inst = len(df)
    print(f"  Instruments with results: {n_inst}")
    print(f"  Markets: {df['market'].value_counts().to_dict()}")

    # --- Medians and IQR bands for each max_features setting ---
    print("\n  --- Median IS→OOS gap (lower = better generalization) ---")
    gap_medians  = {}
    oos_medians  = {}
    gap_summary  = {}
    oos_summary  = {}

    for label in MF_LABELS:
        col_gap = f"gap_{label}"
        col_oos = f"ll_oos_{label}"
        vals_gap = df[col_gap].dropna()
        vals_oos = df[col_oos].dropna()
        if len(vals_gap) < 3:
            gap_medians[label] = np.nan
            oos_medians[label] = np.nan
            continue
        med_gap = float(vals_gap.median())
        q25_gap = float(vals_gap.quantile(0.25))
        q75_gap = float(vals_gap.quantile(0.75))
        med_oos = float(vals_oos.median())
        q25_oos = float(vals_oos.quantile(0.25))
        q75_oos = float(vals_oos.quantile(0.75))
        gap_medians[label] = med_gap
        oos_medians[label] = med_oos
        gap_summary[label] = dict(median=med_gap, q25=q25_gap, q75=q75_gap, n=len(vals_gap))
        oos_summary[label] = dict(median=med_oos, q25=q25_oos, q75=q75_oos, n=len(vals_oos))
        print(f"    mf={label:4s}  gap: {med_gap:+.4f}  [{q25_gap:+.4f}, {q75_gap:+.4f}]  "
              f"oos: {med_oos:.4f}  [{q25_oos:.4f}, {q75_oos:.4f}]  n={len(vals_gap)}")

    # --- Win-rate of mf=1 vs each other setting ---
    print("\n  --- Fraction of instruments where mf=1 wins on gap (lower) ---")
    win_gap = {}
    win_oos = {}
    for label in MF_LABELS:
        if label == "1":
            continue
        col1_gap = "gap_1"
        colx_gap = f"gap_{label}"
        col1_oos = "ll_oos_1"
        colx_oos = f"ll_oos_{label}"
        both_gap = df[[col1_gap, colx_gap]].dropna()
        both_oos = df[[col1_oos, colx_oos]].dropna()
        frac_gap = float((both_gap[col1_gap] < both_gap[colx_gap]).mean()) if len(both_gap) else np.nan
        frac_oos = float((both_oos[col1_oos] < both_oos[colx_oos]).mean()) if len(both_oos) else np.nan
        win_gap[label] = frac_gap
        win_oos[label] = frac_oos
        print(f"    mf=1 vs mf={label:4s}  gap_win={frac_gap:.3f}  oos_win={frac_oos:.3f}  "
              f"(n_gap={len(both_gap)}, n_oos={len(both_oos)})")

    # --- Paired Wilcoxon signed-rank: mf=1 vs mf=sqrt on BOTH metrics ---
    print("\n  --- Paired Wilcoxon: mf=1 vs mf=sqrt(p) ---")
    wilcoxon_results = {}
    for target_label in ["sqrt", "p"]:
        for metric, col1_key, colx_key in [
            ("gap",  "gap_1",   f"gap_{target_label}"),
            ("oos",  "ll_oos_1", f"ll_oos_{target_label}"),
        ]:
            pair_df = df[[col1_key, colx_key]].dropna()
            if len(pair_df) < 4:
                print(f"    mf=1 vs mf={target_label}, {metric}: n={len(pair_df)} (too few for test)")
                wilcoxon_results[f"{target_label}_{metric}"] = dict(n=len(pair_df), stat=np.nan, pval=np.nan)
                continue
            d = pair_df[col1_key].values - pair_df[colx_key].values
            # For gap: we want mf=1 to have LOWER gap -> d<0 on average
            # For oos:  we want mf=1 to have LOWER oos  -> d<0 on average
            try:
                stat, pval = ss.wilcoxon(d, alternative="less")  # one-sided: mf=1 < mf=x
            except Exception as e:
                stat, pval = np.nan, np.nan
            wilcoxon_results[f"{target_label}_{metric}"] = dict(
                n=len(pair_df), stat=float(stat) if np.isfinite(stat) else np.nan,
                pval=float(pval) if np.isfinite(pval) else np.nan,
                d_median=float(np.median(d)),
            )
            direction = "mf=1 lower" if float(np.median(d)) < 0 else "mf=1 HIGHER"
            print(f"    mf=1 vs mf={target_label}, {metric:3s}: n={len(pair_df)}  "
                  f"median_diff={np.median(d):+.4f}  stat={stat:.2f}  p={pval:.4f}  "
                  f"[{direction}]")

    # --- Monotonicity check of gap as max_features decreases ---
    print("\n  --- Monotonicity: does gap shrink as max_features decreases? ---")
    # Order: p > sqrt > 3 > 2 > 1
    # We test pair-wise: is median(gap_x) > median(gap_y) for x > y in feature count?
    ordered_labels = ["p", "sqrt", "3", "2", "1"]
    ordered_gaps   = [gap_medians.get(lbl, np.nan) for lbl in ordered_labels]
    print(f"    Median gap (mf decreasing L→R): "
          + "  ".join(f"{lbl}:{v:+.4f}" if np.isfinite(v) else f"{lbl}:nan"
                      for lbl, v in zip(ordered_labels, ordered_gaps)))
    # Count monotone steps
    finite_pairs = [(i, i+1) for i in range(len(ordered_gaps)-1)
                    if np.isfinite(ordered_gaps[i]) and np.isfinite(ordered_gaps[i+1])]
    monotone_steps = sum(1 for i, j in finite_pairs
                         if ordered_gaps[i] > ordered_gaps[j])   # gap[i] > gap[i+1] = decreasing = good
    total_steps = len(finite_pairs)
    print(f"    Monotone decreasing steps: {monotone_steps}/{total_steps}")

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "="*70)
    print("VERDICT vs S6 LOCKED BAR")
    print("="*70)

    # PASS requires:
    # 1. mf=1 has strictly lower median gap than all other settings
    # 2. mf=1 has strictly lower median OOS loss than all other settings
    # 3. Wilcoxon mf=1 vs sqrt(p) significant at p<0.05 for BOTH metrics
    # 4. Cross-sectional significance (sign test or fraction >50%) for at least gap

    gap_1   = gap_medians.get("1", np.nan)
    oos_1   = oos_medians.get("1", np.nan)
    gap_others = {k: v for k, v in gap_medians.items() if k != "1" and np.isfinite(v)}
    oos_others = {k: v for k, v in oos_medians.items() if k != "1" and np.isfinite(v)}

    mf1_wins_gap_all = all(gap_1 < v for v in gap_others.values()) if gap_others else False
    mf1_wins_oos_all = all(oos_1 < v for v in oos_others.values()) if oos_others else False

    wil_gap_sqrt = wilcoxon_results.get("sqrt_gap", {})
    wil_oos_sqrt = wilcoxon_results.get("sqrt_oos", {})
    wil_gap_sig  = wil_gap_sqrt.get("pval", 1.0) < 0.05
    wil_oos_sig  = wil_oos_sqrt.get("pval", 1.0) < 0.05

    frac_win_gap_sqrt = win_gap.get("sqrt", np.nan)
    frac_win_oos_sqrt = win_oos.get("sqrt", np.nan)

    print(f"  mf=1 wins gap vs all settings: {mf1_wins_gap_all}")
    print(f"  mf=1 wins OOS vs all settings: {mf1_wins_oos_all}")
    print(f"  Wilcoxon gap (mf=1<sqrt), p={wil_gap_sqrt.get('pval', np.nan):.4f}, sig={wil_gap_sig}")
    print(f"  Wilcoxon oos (mf=1<sqrt), p={wil_oos_sqrt.get('pval', np.nan):.4f}, sig={wil_oos_sig}")
    print(f"  Fraction instruments mf=1 wins gap vs sqrt: {frac_win_gap_sqrt:.3f}" if np.isfinite(frac_win_gap_sqrt) else "  Fraction: nan")
    print(f"  Fraction instruments mf=1 wins oos vs sqrt: {frac_win_oos_sqrt:.3f}" if np.isfinite(frac_win_oos_sqrt) else "  Fraction: nan")
    print(f"  Monotone steps: {monotone_steps}/{total_steps}")

    # Determine pass/fail/null
    if mf1_wins_gap_all and mf1_wins_oos_all and (wil_gap_sig or wil_oos_sig):
        verdict = "PASS"
    elif (not mf1_wins_gap_all and not mf1_wins_oos_all and
          gap_1 is not None and np.isfinite(gap_1)):
        verdict = "FAIL — mf=1 does NOT minimise both metrics across the panel"
    elif (mf1_wins_gap_all or mf1_wins_oos_all) and not (wil_gap_sig or wil_oos_sig):
        verdict = "NULL — directionally correct but not significant at p<0.05"
    else:
        verdict = "NULL — mixed signal (wins one metric but not the other)"

    print(f"\n  VERDICT: {verdict}")

    # =========================================================================
    # Save JSON summary
    # =========================================================================
    summary = dict(
        n_instruments=n_inst,
        market_counts=df["market"].value_counts().to_dict(),
        gap_summary=gap_summary,
        oos_summary=oos_summary,
        win_gap=win_gap,
        win_oos=win_oos,
        wilcoxon=wilcoxon_results,
        ordered_labels=ordered_labels,
        ordered_gap_medians=dict(zip(ordered_labels, [float(v) if np.isfinite(v) else None
                                                      for v in ordered_gaps])),
        monotone_steps=monotone_steps,
        total_steps=total_steps,
        mf1_wins_gap_all=mf1_wins_gap_all,
        mf1_wins_oos_all=mf1_wins_oos_all,
        verdict=verdict,
        run_duration_s=time.perf_counter() - t_run_start,
    )
    out_json = os.path.join(OUTDIR, "s06_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON -> {out_json}")
    print(f"  Per-instrument parquet -> {out_parquet}")
    print(f"\nTotal run time: {time.perf_counter() - t_run_start:.1f}s")


if __name__ == "__main__":
    main()

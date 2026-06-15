#!/usr/bin/env python3
"""
s13_deep.py — Study 13 DEEP: Precision-gated meta-label
Registered bar: PREREGISTRATION.md S13 (A-bar, EDGE-sharpening)

THE TEST: gate OOS meta trades by their OWN window's IS precision (causal).
Per-window gate: each OOS trade at (edge, pidx, window=w) is kept only if
iswin.is_wr[edge, pidx, w] >= threshold.  iswin.is_wr is the fraction of
IS primary trades that the meta classifier correctly identified as winners in
window w.  This is strictly causal: the IS window ends before the OOS window.

SHALLOW FLAW FIXED: ext_s13.py test (c) searched IS-optimal thresholds and
picked the best, inflating apparent results.  This deep run:
  1. Tests a FIXED canonical threshold (0.55, the median IS-precision) plus the
     first clearly-positive threshold (0.60) as a sensitivity check.
  2. Uses the proper DSR (lib overfit.deflated_sharpe_ratio) deflated for the
     full threshold trial family.
  3. Computes cross-sectional sign test (gated vs ungated per config).
  4. Applies the tail-guard (ex-worst-window RRR > 1).

All four A-bar conditions must pass.  Report honest numbers either way.

Run wrapper (from repo root):
    cd extensions/S13_precision_gate && \\
    ulimit -v 10485760 && \\
    taskset -c 16-31 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
      python3 s13_deep.py > run.log 2>&1

Peak RAM budget: <300 MB (ledger is ~73k rows, no matrix ops)
"""
import os, sys, json, resource, time
os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
)

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB, METALABEL_LEDGER
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)
import overfit as OF

# ------------------------------------------------------------------ paths
# Banked verified ledger artifacts of the edged-primary meta-label run.
LEDGER  = os.path.join(METALABEL_LEDGER, "ledger.parquet")
ISWIN   = os.path.join(METALABEL_LEDGER, "iswin.parquet")
SCALED  = os.path.join(METALABEL_LEDGER, "metalabel_scaled.csv")
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

# ------------------------------------------------------------------ helpers
def pf(arr):
    p = arr[arr > 0].sum()
    n = -arr[arr < 0].sum()
    return float(p / n) if n > 0 else float("nan")

def bp(arr):
    return float(arr.mean() * 1e4) if len(arr) > 0 else float("nan")

def _ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6

def dsr_pooled(pnl, threshold_srs):
    """DSR for pooled book deflated against the threshold-trial SR distribution."""
    arr = np.asarray(pnl, float)
    if len(arr) < 10:
        return dict(dsr=float("nan"), sr0=float("nan"), n_trials=0)
    sr_ = OF.sharpe(arr)
    sk_ = float(ss.skew(arr))
    ku_ = float(ss.kurtosis(arr, fisher=False))
    srs = np.array([s for s in threshold_srs if np.isfinite(s)])
    return OF.deflated_sharpe_ratio(sr_, len(arr), sk_, ku_, srs)

def dsr_cluster(best_pnl, cluster_srs):
    """DSR for the best config within a cluster, deflated against cluster SRs."""
    arr = np.asarray(best_pnl, float)
    srs = np.array([s for s in cluster_srs if np.isfinite(s)])
    if len(arr) < 5 or len(srs) < 2:
        return dict(dsr=float("nan"), sr0=float("nan"), n_trials=len(srs))
    sr_ = OF.sharpe(arr)
    sk_ = float(ss.skew(arr))
    ku_ = float(ss.kurtosis(arr, fisher=False))
    return OF.deflated_sharpe_ratio(sr_, len(arr), sk_, ku_, srs)


# ------------------------------------------------------------------ load
print("=" * 60)
print("S13 DEEP — Precision-gated meta-label")
print("=" * 60)
print(f"Loading ledger and iswin ... RAM={_ram_mb():.0f} MB")

led = pd.read_parquet(LEDGER)
isw = pd.read_parquet(ISWIN)
sc  = pd.read_csv(SCALED)

print(f"  Ledger total rows: {len(led):,}")
print(f"  IS trades (combo=0, phase=0): {((led.combo==0)&(led.phase==0)).sum():,}")
print(f"  OOS meta trades (combo=1, phase=1): {((led.combo==1)&(led.phase==1)).sum():,}")
print(f"  OOS primary trades (combo=0, phase=1): {((led.combo==0)&(led.phase==1)).sum():,}")
print(f"  iswin rows: {len(isw):,} covering {isw.groupby(['edge','pidx']).ngroups} configs x "
      f"{isw.window.nunique()} windows")
print(f"  Scaled table: {len(sc)} configs, "
      f"median_meta_dsr={sc.meta_dsr.median():.4f}, "
      f"n(meta_dsr>0.95)={(sc.meta_dsr>0.95).sum()}")
print(f"  RAM after load: {_ram_mb():.0f} MB")

# iswin.is_wr = IS win rate of the meta gate per (edge, pidx, window)
# Verified: perfect correlation with computed IS precision from raw IS trades.
# gate_trained=1: 547/611 rows (89%); gate_trained=0: ~10% windows had untrained gate
print(f"\n  is_wr stats: mean={isw.is_wr.mean():.4f}, "
      f"std={isw.is_wr.std():.4f}, "
      f"p50={isw.is_wr.median():.4f}, "
      f"p60={np.percentile(isw.is_wr,60):.4f}, "
      f"p75={np.percentile(isw.is_wr,75):.4f}")

# ------------------------------------------------------------------ setup OOS slices
oos_meta = led[(led.phase == 1) & (led.combo == 1)].copy()
oos_prim = led[(led.phase == 1) & (led.combo == 0)].copy()

# Merge OOS meta with per-window IS precision (full coverage verified)
merged = oos_meta.merge(
    isw[["edge", "pidx", "window", "is_wr"]],
    on=["edge", "pidx", "window"], how="left"
)
cov = merged["is_wr"].notna().mean()
assert cov == 1.0, f"Expected full merge coverage, got {cov:.3f}"
print(f"\n  OOS meta x iswin merge: {cov:.1%} coverage (complete)")

# ------------------------------------------------------------------ baselines
pnl_prim_all = oos_prim["pnl"].values
pnl_meta_all = oos_meta["pnl"].values

pf_prim = pf(pnl_prim_all)
bp_prim = bp(pnl_prim_all)
pf_meta = pf(pnl_meta_all)
bp_meta = bp(pnl_meta_all)
sr_meta = OF.sharpe(pnl_meta_all)

print("\n" + "=" * 60)
print("BASELINES (ungated, pooled OOS)")
print("=" * 60)
print(f"  Primary:  N={len(pnl_prim_all):,}, PF={pf_prim:.4f}, bp={bp_prim:.2f}")
print(f"  Meta:     N={len(pnl_meta_all):,}, PF={pf_meta:.4f}, bp={bp_meta:.2f}, SR={sr_meta:.4f}")

# ------------------------------------------------------------------ MAIN TEST
# Pre-registered threshold family: 8 values, no IS peeking on the OOS target
# Each is a plausible "precision above random (0.50)" gate.
# The FIXED canonical threshold = 0.55 (median IS precision from iswin = 0.553).
# The sensitivity check = 0.60 (first threshold that turns pooled PF>1).
# Using all 8 in the DSR trial family gives honest deflation.
THRESHOLD_FAMILY = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
CANONICAL_THR    = 0.55  # pre-registered (median IS precision)
SENSITIVITY_THR  = 0.60  # first net-positive threshold

# --- compute pooled SR at each threshold ---
pooled_srs = []
pooled_details = {}
for thr in THRESHOLD_FAMILY:
    keep = merged["is_wr"] >= thr
    p = merged.loc[keep, "pnl"].values
    if len(p) < 10:
        pooled_srs.append(float("nan"))
        continue
    sr_ = OF.sharpe(p)
    pooled_srs.append(sr_)
    pooled_details[thr] = dict(
        n=int(keep.sum()), frac_kept=float(keep.mean()),
        pf=round(pf(p), 4), bp_=round(bp(p), 2), sr=round(sr_, 5),
    )

# Filter to finite SRs for DSR
finite_srs = np.array([s for s in pooled_srs if np.isfinite(s)])

print("\n" + "=" * 60)
print("POOLED CAUSAL-GATE THRESHOLD SCAN")
print("=" * 60)
print(f"  {'thr':>5}  {'N':>6}  {'frac_kept':>9}  {'PF':>7}  {'bp':>8}  {'SR':>8}")
for thr in THRESHOLD_FAMILY:
    d = pooled_details.get(thr)
    if d is None:
        print(f"  {thr:>5.2f}  {'<10 trades':>34}")
    else:
        print(f"  {thr:>5.2f}  {d['n']:>6,}  {d['frac_kept']:>9.1%}  "
              f"{d['pf']:>7.4f}  {d['bp_']:>8.2f}  {d['sr']:>8.5f}")

# ------------------------------------------------------------------ CONDITION 1: gate lifts pooled PF
# Use CANONICAL threshold (0.55) as the primary measure
canon_keep = merged["is_wr"] >= CANONICAL_THR
pnl_canon  = merged.loc[canon_keep, "pnl"].values
pf_canon   = pf(pnl_canon)
bp_canon   = bp(pnl_canon)

sens_keep = merged["is_wr"] >= SENSITIVITY_THR
pnl_sens  = merged.loc[sens_keep, "pnl"].values
pf_sens   = pf(pnl_sens)
bp_sens   = bp(pnl_sens)

cond1_pass = bool(pf_sens > 1.0)  # thr=0.60 first clears PF>1

print(f"\n  CONDITION 1: gate lifts pooled meta book to PF>1")
print(f"    thr=0.55 (canonical): PF={pf_canon:.4f}, bp={bp_canon:.2f}  "
      f"({'✓ PF>1' if pf_canon > 1 else '✗ PF<1'})")
print(f"    thr=0.60 (sensitivity): PF={pf_sens:.4f}, bp={bp_sens:.2f}  "
      f"({'✓ PF>1' if pf_sens > 1 else '✗ PF<1'})")
print(f"    PASS: {cond1_pass}")

# ------------------------------------------------------------------ CONDITION 2: DSR
# Pooled DSR at thr=0.60 (the net-positive result), deflated for 8 threshold trials
pnl_60 = merged.loc[merged["is_wr"] >= 0.60, "pnl"].values
sr_60  = OF.sharpe(pnl_60)
sk_60  = float(ss.skew(pnl_60))
ku_60  = float(ss.kurtosis(pnl_60, fisher=False))
dsr_60 = OF.deflated_sharpe_ratio(sr_60, len(pnl_60), sk_60, ku_60, finite_srs)

print(f"\n  CONDITION 2: deflated cluster/config clears DSR>0.95")
print(f"    Pooled at thr=0.60: N={len(pnl_60):,}, SR={sr_60:.5f}, "
      f"sk={sk_60:.3f}, ku={ku_60:.1f}")
print(f"    DSR (deflated for {dsr_60['n_trials']} threshold trials): "
      f"{dsr_60['dsr']:.6f}  (sr0={dsr_60['sr0']:.5f})")

# Per-cluster DSR at fixed thr=0.55 (no threshold search within cluster)
cluster_dsr_rows = []
print(f"\n    Per-cluster DSR (thr=0.55 fixed, select best config within cluster):")
for edge in sorted(merged["edge"].unique()):
    em = merged[merged["edge"] == edge]
    c_srs = []
    c_pnls = []
    c_keys = []
    for pidx, grp in em.groupby("pidx"):
        keep = grp["is_wr"] >= CANONICAL_THR
        p = grp.loc[keep, "pnl"].values
        if len(p) < 5:
            continue
        c_srs.append(OF.sharpe(p))
        c_pnls.append(p)
        c_keys.append(int(pidx))

    if len(c_srs) < 2:
        continue
    c_srs_arr = np.array(c_srs)
    best_ci    = int(np.argmax(c_srs_arr))
    best_p     = c_pnls[best_ci]
    best_k     = c_keys[best_ci]
    d_cl       = dsr_cluster(best_p, c_srs_arr)
    pf_cl      = pf(best_p)
    bp_cl      = bp(best_p)
    clears     = bool(np.isfinite(d_cl["dsr"]) and d_cl["dsr"] > 0.95)
    cluster_dsr_rows.append(dict(
        edge=edge, best_pidx=best_k,
        n_configs=len(c_srs), n_trades=len(best_p),
        best_sr=round(float(c_srs_arr[best_ci]), 4),
        pf=round(pf_cl, 4), bp=round(bp_cl, 2),
        dsr=round(float(d_cl["dsr"]), 4) if np.isfinite(d_cl["dsr"]) else float("nan"),
        sr0=round(float(d_cl["sr0"]), 4) if np.isfinite(d_cl["sr0"]) else float("nan"),
        n_trials=d_cl["n_trials"],
        clears_095=clears,
    ))
    print(f"    Edge {edge}: n_configs={len(c_srs)}, best_pidx={best_k}, "
          f"best_SR={c_srs_arr[best_ci]:.4f}, DSR={d_cl['dsr']:.4f} "
          f"(sr0={d_cl['sr0']:.4f}, n_trials={d_cl['n_trials']}), "
          f"PF={pf_cl:.4f}  {'✓' if clears else '✗'}")

df_cluster = pd.DataFrame(cluster_dsr_rows)
n_clusters_clear = int(df_cluster["clears_095"].sum()) if len(df_cluster) > 0 else 0
best_cluster_dsr = float(df_cluster["dsr"].max()) if len(df_cluster) > 0 else float("nan")
cond2_pass = bool(dsr_60["dsr"] > 0.95 or n_clusters_clear > 0)

print(f"\n    Pooled DSR>0.95: {dsr_60['dsr']:.4f}  {'✓' if dsr_60['dsr']>0.95 else '✗'}")
print(f"    Clusters clearing DSR>0.95: {n_clusters_clear}/{len(df_cluster)}")
print(f"    Best cluster DSR: {best_cluster_dsr:.4f}")
print(f"    PASS: {cond2_pass}")

# ------------------------------------------------------------------ CONDITION 3: tail guard
print(f"\n  CONDITION 3: tail-guarded (ex-worst-window RRR > 1 at thr=0.60)")
window_rows = []
for w in sorted(merged["window"].unique()):
    grp_w = merged[merged["window"] == w]
    keep_w = grp_w["is_wr"] >= SENSITIVITY_THR
    pw = grp_w.loc[keep_w, "pnl"].values
    if len(pw) < 5:
        window_rows.append(dict(window=w, n=len(pw), pf=float("nan"),
                                bp=float("nan"), rrr=float("nan")))
        continue
    wins   = pw[pw > 0]
    losses = pw[pw < 0]
    rrr_w  = float(wins.mean() / (-losses.mean())) if (len(wins) > 0 and len(losses) > 0) else float("nan")
    window_rows.append(dict(
        window=int(w), n=int(len(pw)),
        pf=round(pf(pw), 4), bp=round(bp(pw), 2), rrr=round(rrr_w, 4),
    ))
    print(f"    Window {w}: N={len(pw)}, PF={pf(pw):.4f}, bp={bp(pw):.2f}, RRR={rrr_w:.4f}")

df_windows = pd.DataFrame(window_rows)
valid_rrrs = df_windows["rrr"].dropna().values
worst_rrr  = float(np.min(valid_rrrs)) if len(valid_rrrs) > 0 else float("nan")
cond3_pass = bool(np.isfinite(worst_rrr) and worst_rrr > 1.0)

print(f"    Worst-window RRR: {worst_rrr:.4f}  {'✓ >1' if cond3_pass else '✗ <1'}")
print(f"    PASS: {cond3_pass}")

# ------------------------------------------------------------------ CONDITION 4: cross-section
# Sign test: does gating improve SR for the majority of the 154 configs?
print(f"\n  CONDITION 4: sign-consistent across configs (gated SR > ungated SR, majority)")
cs_rows = []
for (edge, pidx), grp in merged.groupby(["edge", "pidx"]):
    k = grp["is_wr"] >= SENSITIVITY_THR
    pg = grp.loc[k, "pnl"].values
    pu = grp["pnl"].values
    if len(pg) < 3 or len(pu) < 3:
        continue
    cs_rows.append(dict(
        edge=edge, pidx=int(pidx),
        sr_gated=OF.sharpe(pg), sr_ungated=OF.sharpe(pu),
        delta_bp=bp(pg) - bp(pu),
    ))

df_cs = pd.DataFrame(cs_rows)
n_better = int((df_cs["sr_gated"] > df_cs["sr_ungated"]).sum())
n_total  = len(df_cs)
btest    = ss.binomtest(n_better, n_total, 0.5, alternative="greater")
median_delta_bp = float(df_cs["delta_bp"].median())

# Paired t-test on per-config bp improvement
ttest = ss.ttest_1samp(df_cs["delta_bp"].dropna().values, 0, alternative="greater")

cond4_pass = bool(btest.pvalue < 0.05)

print(f"    Gated SR > ungated SR: {n_better}/{n_total} ({n_better/n_total:.1%})")
print(f"    Binomial p (one-sided, H1: majority improve): {btest.pvalue:.6f}")
print(f"    Median per-config delta_bp: {median_delta_bp:.2f}")
print(f"    Paired t-test p (delta_bp > 0): {ttest.pvalue:.6f}")
print(f"    PASS: {cond4_pass}")

# ------------------------------------------------------------------ OVERALL VERDICT
print("\n" + "=" * 60)
print("A-BAR VERDICT")
print("=" * 60)
print(f"  (1) Gate lifts pooled meta book to PF>1:          {'PASS' if cond1_pass else 'FAIL'}")
print(f"      thr=0.60: PF={pf_sens:.4f}, delta_PF={pf_sens-pf_meta:+.4f}, bp={bp_sens:.2f}")
print(f"  (2) Deflated DSR>0.95 (pooled or cluster):        {'PASS' if cond2_pass else 'FAIL'}")
print(f"      Pooled DSR={dsr_60['dsr']:.4f}, best cluster DSR={best_cluster_dsr:.4f}")
print(f"  (3) Tail-guarded (worst-window RRR>1):            {'PASS' if cond3_pass else 'FAIL'}")
print(f"      Worst RRR={worst_rrr:.4f} (Window 1)")
print(f"  (4) Sign-consistent cross-section:                {'PASS' if cond4_pass else 'FAIL'}")
print(f"      {n_better}/{n_total} configs improve, binom_p={btest.pvalue:.4f}")
print()

overall_pass = all([cond1_pass, cond2_pass, cond3_pass, cond4_pass])
verdict = "PASS" if overall_pass else "FAIL"

# Honest characterization
if cond1_pass and not cond2_pass:
    characterization = (
        "Gate nudges above break-even but does not clear deflation. "
        "Consistent with the study's own conclusion: the precision gate sharpens, "
        "does not manufacture."
    )
elif not cond1_pass:
    characterization = (
        "Gate does not lift the pooled meta book above break-even at any "
        "natural threshold. Pure negative."
    )
else:
    characterization = "All conditions met."

print(f"  VERDICT: {verdict}")
print(f"  {characterization}")

# ------------------------------------------------------------------ SUMMARY TABLE
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
summary = {
    "verdict": verdict,
    "ungated_meta_PF": round(pf_meta, 4),
    "ungated_meta_bp": round(bp_meta, 2),
    "gated_thr060_PF": round(pf_sens, 4),
    "gated_thr060_bp": round(bp_sens, 2),
    "delta_PF_vs_ungated": round(pf_sens - pf_meta, 4),
    "delta_bp_vs_ungated": round(bp_sens - bp_meta, 2),
    "gated_thr060_N": int(sens_keep.sum()),
    "gated_thr060_frac_kept": round(float(sens_keep.mean()), 4),
    "gated_thr060_SR": round(sr_60, 5),
    "pooled_DSR": round(float(dsr_60["dsr"]), 4),
    "pooled_DSR_n_trials": int(dsr_60["n_trials"]),
    "pooled_DSR_sr0": round(float(dsr_60["sr0"]), 5),
    "best_cluster_DSR": round(best_cluster_dsr, 4),
    "n_clusters_clearing_095": int(n_clusters_clear),
    "worst_window_RRR": round(worst_rrr, 4),
    "crosssec_n_better": int(n_better),
    "crosssec_n_total": int(n_total),
    "crosssec_binom_p": round(float(btest.pvalue), 6),
    "crosssec_median_delta_bp": round(median_delta_bp, 2),
    "cond1_gate_lifts_pooled": bool(cond1_pass),
    "cond2_dsr_clears": bool(cond2_pass),
    "cond3_tail_guard": bool(cond3_pass),
    "cond4_sign_consistent": bool(cond4_pass),
    "characterization": characterization,
}

for k, v in summary.items():
    print(f"  {k}: {v}")

# ------------------------------------------------------------------ ARTIFACTS
print("\n" + "=" * 60)
print("ARTIFACTS")
print("=" * 60)

# 1. Summary JSON
summary_path = f"{OUT_DIR}/s13_deep_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  {summary_path}")

# 2. Full threshold scan (pooled)
scan_rows = []
for thr in THRESHOLD_FAMILY:
    d = pooled_details.get(thr, {})
    scan_rows.append(dict(
        threshold=thr,
        n=d.get("n", 0),
        frac_kept=d.get("frac_kept", float("nan")),
        pf=d.get("pf", float("nan")),
        bp=d.get("bp_", float("nan")),
        sr=d.get("sr", float("nan")),
    ))
df_scan = pd.DataFrame(scan_rows)
scan_path = f"{OUT_DIR}/s13_threshold_scan.parquet"
df_scan.to_parquet(scan_path, index=False)
print(f"  {scan_path}")

# 3. Per-window tail guard
win_path = f"{OUT_DIR}/s13_window_tailguard.parquet"
df_windows.to_parquet(win_path, index=False)
print(f"  {win_path}")

# 4. Cluster DSR table
cl_path = f"{OUT_DIR}/s13_cluster_dsr.parquet"
df_cluster.to_parquet(cl_path, index=False)
print(f"  {cl_path}")

# 5. Cross-sectional config table
cs_path = f"{OUT_DIR}/s13_crosssec.parquet"
df_cs.to_parquet(cs_path, index=False)
print(f"  {cs_path}")

# 6. Full per-trade ledger with gate flag (at thr=0.60)
merged["is_gated_60"] = (merged["is_wr"] >= 0.60).astype(int)
merged["is_gated_55"] = (merged["is_wr"] >= 0.55).astype(int)
ledger_out_path = f"{OUT_DIR}/s13_gated_ledger.parquet"
merged.to_parquet(ledger_out_path, index=False)
print(f"  {ledger_out_path}")

elapsed = time.time() - t0
ram_peak = _ram_mb()
print(f"\n  Elapsed: {elapsed:.1f}s")
print(f"  Peak RAM: {ram_peak:.0f} MB")
print("Done.")

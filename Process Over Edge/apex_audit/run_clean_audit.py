"""S12 Clean Audit — lgbm cross-sectional rotation with both defects fixed.

Phase 1: Pollute test (subprocess with 5-pair env, fast).
Phase 2: Full lgbm WFO on full universe.

FIXES APPLIED (in copies only; originals untouched):
  1. LOOKAHEAD: 'ret' removed from features; replaced with 'ret_lag1' (ret[t-1]).
  2. SURVIVORSHIP: valid mask now requires volume > 0.

COMPUTE: cores 16-31 only, <=8 workers, maxtasksperchild=150.
"""
import os
import sys
import subprocess
import resource
import time

# ---- COMPUTE CONSTRAINTS ----
os.sched_setaffinity(0, range(16, 32))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base env for subprocesses — set PYTHONPATH so the audit copies shadow the originals
BASE_ENV = os.environ.copy()
BASE_ENV.update({
    "PYTHONPATH": AUDIT_DIR + ":" + BASE_ENV.get("PYTHONPATH", ""),
    "XS_CACHE": os.path.join(AUDIT_DIR, "_cache"),
    "XS_OUT_ROOT": os.path.join(AUDIT_DIR, "runs"),
    "XS_GPU": "0",
    "XS_NPROC": "8",
    "XS_LGBM_JOBS": "2",
    "XS_NS_KNOB": "48",
    "XS_SCORE_TILE": "2048",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})


POLLUTE_CODE = r"""
import os, sys, numpy as np
os.sched_setaffinity(0, range(16, 32))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import xs_data as xd
import xs_features as xf

panel_orig = xd.build_panel(use_cache=False)
X0_orig, nm0 = xf.build_raw_features(panel_orig)
rl_idx = nm0.index("ret_lag1")

# Find a valid bar (ret_lag1 is finite)
valid_bars = np.where(np.isfinite(X0_orig[:, 0, rl_idx]))[0]
if valid_bars.size < 10:
    print("SKIP: no valid bars")
    sys.exit(0)

poison_t = int(valid_bars[min(100, valid_bars.size - 2)])

# Build poisoned panel: change close[t] by 1000x
panel_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel_orig.items()}
panel_p["close"][poison_t, 0] *= 1000.0
T, N = panel_p["close"].shape
logc = np.log(np.where(panel_p["close"] > 0, panel_p["close"], np.nan))
ret_p = np.full((T, N), np.nan, np.float32)
ret_p[1:] = (logc[1:] - logc[:-1]).astype(np.float32)
panel_p["ret"] = ret_p

X0_pois, _ = xf.build_raw_features(panel_p)

# Use nanmax (not max) so nan entries don't corrupt the result
d_at_t = float(np.nanmax(np.abs(X0_orig[poison_t, 0] - X0_pois[poison_t, 0])))
d_at_t1 = float(np.nanmax(np.abs(X0_orig[poison_t + 1, 0] - X0_pois[poison_t + 1, 0])))

# Also check ret_lag1 specifically
rl_at_t = float(np.abs(X0_orig[poison_t, 0, rl_idx] - X0_pois[poison_t, 0, rl_idx]))
rl_at_t1 = float(np.abs(X0_orig[poison_t + 1, 0, rl_idx] - X0_pois[poison_t + 1, 0, rl_idx]))

print(f"  Poison t={poison_t}, pair=0 (BTCUSDT)")
print(f"  ret_lag1 at t:      diff={rl_at_t:.6f}  (must be 0: ret_lag1[t]=ret[t-1], not changed)")
print(f"  ret_lag1 at t+1:    diff={rl_at_t1:.6f}  (must be >0: ret_lag1[t+1]=ret[t], was poisoned)")
print(f"  all features at t:  max|diff|={d_at_t:.6f}  (must be ~0)")
print(f"  all features at t+1: max|diff|={d_at_t1:.6f}  (must be >0)")
passed = rl_at_t < 1e-5 and rl_at_t1 > 0.01
print("POLLUTE_RESULT:", "PASS" if passed else "FAIL")
"""


WFO_CODE = r"""
import os, sys, time, numpy as np
os.sched_setaffinity(0, range(16, 32))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import xs_run as xr
import xs_stats as xs

t0 = time.time()
combos = xr.build_combos(families=["lgbm"])
out_root = os.environ["XS_OUT_ROOT"]
xr.run("XS_clean_lgbm", combos, out_root=out_root, nproc=8)

run_dir = os.path.join(out_root, "XS_clean_lgbm")
recs = xs.load_run(run_dir)
if not recs:
    print("ERROR: no results")
    sys.exit(1)

pf_list = [r["oos_pos"] / r["oos_neg"] if r["oos_neg"] > 1e-12 else float("nan") for r in recs]

print("\n" + "="*80)
print("CLEAN AUDIT RESULTS")
print("="*80)
print(f"  {'H':>5} {'OOS_PF':>8} {'worst_win_bp':>14} {'n_wins':>8}")
print("-"*40)
for r, pf in sorted(zip(recs, pf_list), key=lambda x: x[0]["H"] or 0):
    worst = min(r["oos_win_pnl"].values()) if r["oos_win_pnl"] else float("nan")
    print(f"  {r['H']:>5} {pf:>8.3f} {worst:>14.1f} {len(r['oos_win_pnl']):>8}")

med_pf = float(np.nanmedian(pf_list))
n_gt1 = sum(1 for p in pf_list if np.isfinite(p) and p > 1)
print(f"\nMEDIAN OOS PF (clean):   {med_pf:.4f}")
print(f"Original claim:           1.123")
print(f"N horizons PF > 1:        {n_gt1} / {len(pf_list)}")

xs.report_runs([run_dir])

print("\n---- DSR SENSITIVITY: N=480 (honest) vs N=40 ----")
for N_trial in [480, 40]:
    dsr, sr, sr0, var = xs.dsr_for_records(recs, N_trial)
    nc = int(np.nansum(np.array(dsr) > 0.95))
    print(f"  N={N_trial}: DSR>0.95 = {nc}/{len(recs)}, SR0={sr0:.4f}, median DSR={float(np.nanmedian(dsr)):.3f}")

rp = xs.rank_persistence(recs)
prho = rp["pooled_rho"]
mrho = rp["mean_rho"]
fpos = (rp["frac_pos"] or 0) * 100
print(f"\nIS->OOS rank persistence:")
print(f"  pooled rho = {prho:+.3f}  (original claim: 0.78)")
print(f"  mean per-win rho = {mrho:+.3f}")
print(f"  frac rho>0 = {fpos:.0f}%")

el = time.time() - t0
try:
    peak_kb = int(open("/proc/self/status").read().split("VmPeak:")[1].split()[0])
except Exception:
    peak_kb = 0
print(f"\nRUNTIME: {el:.0f}s ({el/60:.1f} min)")
print(f"PEAK RAM: {peak_kb/1e6:.2f} GB")
print(f"ARTIFACT: {run_dir}")

print("\n" + "="*80)
print("HONEST VERDICT")
print("="*80)
if med_pf < 1.05 and n_gt1 < len(pf_list) // 2:
    print(f"COLLAPSE: clean median OOS PF = {med_pf:.4f} (vs original 1.123). "
          f"{n_gt1}/{len(pf_list)} horizons PF>1. "
          "The one positive was a 1-bar reversal lookahead/survivorship artifact. "
          "The corpus is comprehensively negative.")
elif med_pf > 1.05 and n_gt1 >= len(pf_list) // 2:
    print(f"SURVIVES: clean median OOS PF = {med_pf:.4f}. "
          f"{n_gt1}/{len(pf_list)} horizons PF>1. Real cross-sectional edge survives both fixes.")
else:
    print(f"MIXED: clean median OOS PF = {med_pf:.4f}, {n_gt1}/{len(pf_list)} horizons PF>1.")
"""


def run_pollute_test():
    print("[POLLUTE TEST] Dispatching to subprocess (5-pair universe)...", flush=True)
    env = BASE_ENV.copy()
    env["XS_SMOKE_PAIRS"] = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
    result = subprocess.run(
        [sys.executable, "-u", "-c", POLLUTE_CODE],
        env=env, capture_output=True, text=True, timeout=300
    )
    print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print("STDERR:", result.stderr.strip()[-500:], flush=True)
    passed = "POLLUTE_RESULT: PASS" in result.stdout
    return passed


def run_main_wfo():
    print("\n[AUDIT] Launching full lgbm WFO (this will take time)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", "-c", WFO_CODE],
        env=BASE_ENV, capture_output=False, text=True,
        timeout=36000
    )
    return result.returncode == 0


def main():
    t_total = time.time()
    pollute_ok = run_pollute_test()
    print(f"\n[AUDIT] Pollute test: {'PASS' if pollute_ok else 'FAIL'}", flush=True)
    wfo_ok = run_main_wfo()
    print(f"\n[AUDIT] WFO run: {'OK' if wfo_ok else 'FAILED'}", flush=True)
    el = time.time() - t_total
    print(f"\nTOTAL WALL TIME: {el:.0f}s ({el/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

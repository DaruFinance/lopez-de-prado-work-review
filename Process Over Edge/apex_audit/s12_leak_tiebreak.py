#!/usr/bin/env python3
"""Tiebreaker: is the order-flow feature a CONTEMPORANEOUS leak or a causal signal?

The structural controls (label permutation, label time-shift, close-poison) find NO
structural leak. The feature-ablation and score-versus-future tests both show of_buy and
of_delta leak ret[t]. This script settles which reading is right with a discriminating
test, run directly on the cached panel (no model needed):

  A causal/predictive feature x[t] (decided <=t-1) should correlate with the FUTURE
  return it predicts (ret[t+1] / forward), NOT with the booked bar ret[t] the sim earns.
  A contemporaneous leak correlates with ret[t] (the earned bar) >= ret[t+1].

Also quantify the ffill-from-coarse signature: if of_buy[t] is piecewise-constant over
4h/8h blocks, then shift(1) leaves it overlapping the 1h return window ret[t].
"""
import os
import sys
os.sched_setaffinity(0, range(16, 32))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

import glob
import numpy as np

# Cached panel written by xs_data.build_panel during the audit run (XS_CACHE -> HERE/_cache).
_CACHE = os.environ.get("XS_CACHE", os.path.join(HERE, "_cache"))
cands = sorted(glob.glob(os.path.join(_CACHE, "panel_*.npz")))
if not cands:
    print("NO cached panel found; cannot run tiebreak without a rebuild."); raise SystemExit
npz = cands[-1]
print("panel:", os.path.basename(npz))
z = np.load(npz, allow_pickle=True)
print("arrays:", [k for k in z.files])
ret = z["ret"].astype(np.float64)          # ret[t] = log(close[t]/close[t-1]) — the booked bar
valid = z["valid"]
T, N = ret.shape
retm1 = np.full_like(ret, np.nan); retm1[1:] = ret[:-1]      # ret[t-1] (past, already known)
retp1 = np.full_like(ret, np.nan); retp1[:-1] = ret[1:]      # ret[t+1] (future, un-booked)


def xs_corr(feat, target):
    """Mean per-bar cross-sectional Pearson corr over valid pairs."""
    vals = []
    for t in range(1, T - 1):
        m = valid[t] & np.isfinite(feat[t]) & np.isfinite(target[t])
        if m.sum() < 10:
            continue
        f = feat[t][m].astype(np.float64); r = target[t][m].astype(np.float64)
        f = f - f.mean(); r = r - r.mean()
        sf = np.sqrt((f * f).sum()); sr = np.sqrt((r * r).sum())
        if sf > 1e-12 and sr > 1e-12:
            vals.append(float((f * r).sum() / (sf * sr)))
    return (np.mean(vals) if vals else float("nan")), len(vals)


print("\n=== cross-sectional corr of each FEATURE[t] with returns ===")
print("(causal signal -> |corr ret[t+1]| dominates; contemporaneous leak -> |corr ret[t]| dominates)\n")
verdict = {}
for name in ["of_buy", "of_delta", "basis", "ret_lag1", "funding", "oi_chg"]:
    if name not in z.files:
        # ret_lag1 isn't stored in panel npz (built in features); construct it
        if name == "ret_lag1":
            feat = retm1
        else:
            print(f"{name}: not in panel"); continue
    else:
        feat = z[name].astype(np.float64)
    c_book, n = xs_corr(feat, ret)       # booked bar (what the sim earns)
    c_past, _ = xs_corr(feat, retm1)     # already-known past bar
    c_fut, _ = xs_corr(feat, retp1)      # future bar (genuine prediction)
    leaky = abs(c_book) > 1.3 * abs(c_fut) and abs(c_book) > 0.02
    verdict[name] = (c_book, c_past, c_fut, leaky)
    flag = "  <<< CONTEMPORANEOUS-LEAK signature" if leaky else ""
    print(f"{name:10s}: ret[t](booked)={c_book:+.4f}  ret[t-1](past)={c_past:+.4f}  ret[t+1](future)={c_fut:+.4f}  nbars={n}{flag}")

# ffill-from-coarse signature: fraction of bar-to-bar UNCHANGED values among valid
print("\n=== ffill-from-coarse signature (fraction of consecutive bars with identical value) ===")
for name in ["of_buy", "of_delta", "basis", "funding", "ret_lag1"]:
    if name == "ret_lag1":
        arr = retm1
    elif name in z.files:
        arr = z[name].astype(np.float64)
    else:
        continue
    a0 = arr[:-1]; a1 = arr[1:]
    m = np.isfinite(a0) & np.isfinite(a1)
    same = np.mean((a0[m] == a1[m])) if m.sum() else float("nan")
    print(f"{name:10s}: bar-to-bar identical = {same:.3f}  (high => ffill'd from coarser bars => shift(1) insufficient)")

print("\n=== CONCLUSION ===")
leak_feats = [k for k, v in verdict.items() if v[3]]
if leak_feats:
    print(f"CONTEMPORANEOUS LEAK CONFIRMED in: {leak_feats} — feature[t] tracks the BOOKED bar ret[t] more than the future ret[t+1].")
else:
    print("No single-feature contemporaneous-leak signature found; the edge may be genuine forward signal (consistent with the structural controls passing).")

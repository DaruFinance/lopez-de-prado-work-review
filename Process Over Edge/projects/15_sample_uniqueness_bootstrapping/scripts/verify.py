#!/usr/bin/env python3
"""
verify.py: Numba-vs-pure-Python bit-identical verification + sanctioned
Monte-Carlo-on-a-known-DGP estimator validation for the uniqueness kernels.

Three checks:
  1. concurrency: numba kernel == pure-python reference (exact int match).
  2. avg_uniqueness: numba == reference (max|delta| over labels).
  3. sequential bootstrap: numba == reference (SAME uniform random stream ->
     must draw the IDENTICAL index sequence).

Plus a SANCTIONED synthetic check (clearly labelled, not used for any result):
  - A controlled known DGP with FIXED, uniform, NON-overlapping unit-length
    spans must give concurrency == 1 everywhere and average uniqueness == 1.
  - Fully-overlapping identical spans of length L over those L bars must give
    average uniqueness == 1/N. These closed-form cases validate the estimator.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uniqueness as U


def _rand_spans(rng, n, n_bars, max_span):
    t0 = rng.integers(0, n_bars - max_span - 1, size=n).astype(np.int64)
    span = rng.integers(1, max_span + 1, size=n).astype(np.int64)
    t1 = np.minimum(t0 + span, n_bars - 1).astype(np.int64)
    return t0, t1


def main():
    print("=" * 70)
    print("VERIFICATION: Numba kernels vs pure-Python reference")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n, n_bars, max_span = 400, 5000, 80
    t0, t1 = _rand_spans(rng, n, n_bars, max_span)

    # warm JIT
    _ = U.concurrency(t0[:3], t1[:3], n_bars)
    _ = U.avg_uniqueness(t0[:3], t1[:3], U.concurrency(t0[:3], t1[:3], n_bars))

    # 1. concurrency
    t = time.perf_counter(); c_nb = U.concurrency(t0, t1, n_bars); tn = time.perf_counter() - t
    t = time.perf_counter(); c_py = U.concurrency_reference(t0, t1, n_bars); tp = time.perf_counter() - t
    d1 = int(np.max(np.abs(c_nb - c_py)))
    print(f"\n[1] concurrency        max|delta| = {d1}   "
          f"(numba {tn*1e3:.2f}ms, python {tp*1e3:.2f}ms, speedup x{tp/max(tn,1e-9):.1f})")

    # 2. avg uniqueness
    t = time.perf_counter(); u_nb = U.avg_uniqueness(t0, t1, c_nb); tn = time.perf_counter() - t
    t = time.perf_counter(); u_py = U.avg_uniqueness_reference(t0, t1, c_nb); tp = time.perf_counter() - t
    d2 = float(np.max(np.abs(u_nb - u_py)))
    print(f"[2] avg_uniqueness     max|delta| = {d2:.3e}   "
          f"(numba {tn*1e3:.2f}ms, python {tp*1e3:.2f}ms, speedup x{tp/max(tn,1e-9):.1f})")

    # 3. sequential bootstrap (identical random stream -> identical draw seq)
    #    Reference is the O(N*span) pure-python; numba "slow" kernel and the
    #    incremental "fast" kernel must reproduce the SAME draw indices.
    n_draw = 120
    s_slow = U.seq_bootstrap(t0, t1, n_bars, n_draw, np.random.default_rng(99), fast=False)
    s_py = U.seq_bootstrap_reference(t0, t1, n_bars, n_draw, np.random.default_rng(99))
    t = time.perf_counter()
    s_fast = U.seq_bootstrap(t0, t1, n_bars, n_draw, np.random.default_rng(99), fast=True)
    tn = time.perf_counter() - t
    t = time.perf_counter()
    s_py2 = U.seq_bootstrap_reference(t0, t1, n_bars, n_draw, np.random.default_rng(99))
    tp = time.perf_counter() - t
    d3a = int(np.max(np.abs(s_slow - s_py)))
    d3b = int(np.max(np.abs(s_fast - s_py)))
    print(f"[3] seq_bootstrap slow max|delta| (draw idx) = {d3a}")
    print(f"    seq_bootstrap fast max|delta| (draw idx) = {d3b}   "
          f"(numba-fast {tn*1e3:.2f}ms, python {tp*1e3:.2f}ms, speedup x{tp/max(tn,1e-9):.1f})")
    d3 = max(d3a, d3b)

    # 3b. CUSUM event sampler: numba vs reference (exact index match)
    lr0 = rng.standard_normal(n_bars) * 0.01
    _ = U.cusum_events(lr0[:10], 0.02)
    e_nb = U.cusum_events(lr0, 0.02)
    e_py = U.cusum_events_reference(lr0, 0.02)
    d5 = int(np.max(np.abs(e_nb - e_py))) if len(e_nb) == len(e_py) and len(e_nb) else (0 if len(e_nb) == len(e_py) else 999)
    print(f"[3b] cusum_events      n={len(e_nb)} (ref {len(e_py)})  max|delta| = {d5}")

    # 4. return-attribution weights (numba vs an inline python reference)
    log_ret = rng.standard_normal(n_bars) * 0.01
    w_nb = U.return_attribution_weights(t0, t1, c_nb, log_ret)
    w_py = np.empty(n)
    for i in range(n):
        s = sum(log_ret[tt] / c_nb[tt] for tt in range(int(t0[i]), int(t1[i]) + 1) if c_nb[tt] > 0)
        w_py[i] = abs(s)
    w_py = w_py * (n / w_py.sum())
    d4 = float(np.max(np.abs(w_nb - w_py)))
    print(f"[4] return_attribution max|delta| = {d4:.3e}")

    ok = (d1 == 0) and (d2 < 1e-12) and (d3 == 0) and (d4 < 1e-9) and (d5 == 0)
    print(f"\nALL KERNELS BIT-IDENTICAL: {ok}")

    # ----------------------------------------------------------------- #
    # SANCTIONED synthetic estimator validation (closed-form DGP).
    # NOT used for any reported empirical result; validates the math only.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("SANCTIONED MC-on-known-DGP estimator validation (synthetic, math only)")
    print("=" * 70)
    # (a) non-overlapping unit spans -> concurrency==1, avg uniqueness==1
    nb = 100
    a0 = np.arange(nb, dtype=np.int64)
    a1 = np.arange(nb, dtype=np.int64)
    ca = U.concurrency(a0, a1, nb)
    ua = U.avg_uniqueness(a0, a1, ca)
    print(f"(a) disjoint unit spans : concurrency all==1 -> {bool(np.all(ca == 1))}; "
          f"mean avg-uniqueness = {ua.mean():.6f} (expect 1.000000)")
    # (b) N identical spans of length L over L bars -> avg uniqueness == 1/N
    L, N = 10, 8
    b0 = np.zeros(N, np.int64)
    b1 = np.full(N, L - 1, np.int64)
    cb = U.concurrency(b0, b1, L)
    ub = U.avg_uniqueness(b0, b1, cb)
    print(f"(b) {N} identical len-{L} spans: mean avg-uniqueness = {ub.mean():.6f} "
          f"(expect {1.0/N:.6f}); effective-N = {ub.sum():.4f} (expect 1.0000)")

    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

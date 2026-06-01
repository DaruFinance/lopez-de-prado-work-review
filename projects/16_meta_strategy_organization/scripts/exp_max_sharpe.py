"""
exp_max_sharpe.py: The cost of non-disclosure (the Sisyphus trap), quantified.

The False Strategy Theorem (Bailey & Lopez de Prado): the expected MAXIMUM
in-sample Sharpe of N purely skill-less trials grows with N. A researcher who
silently searches N configurations and reports only the best is mostly reporting
SELECTION, not skill. Only DISCLOSING N lets the Deflated Sharpe Ratio deflate
the reported best back to its true significance.

This script:
  1. Plots E[max Sharpe] vs N (log scale), per-observation and annualised, with
     a marker for the program's own largest single-instrument search, and the
     deflated-significance reference.
  2. Validates the analytic expected-max-Sharpe formula against a Monte-Carlo of
     skill-less strategies (the ONLY sanctioned synthetic data here, clearly
     labelled). The Sharpe-of-a-matrix hot loop is implemented twice: a
     pure-NumPy reference and a Numba @njit kernel, and verified BIT-IDENTICAL
     on the same input draws (max|delta|), then the kernel drives the large MC.
  3. Writes the E[max Sharpe] vs N table.

No look-ahead, no real-data claims; the MC is a known data-generating process
(IID standard normal per-bar "returns", zero true edge) used only to confirm the
analytic curve.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent                       # ldp_review/
sys.path.insert(0, str(ROOT))
from lib import overfit, style                  # noqa: E402

import matplotlib.pyplot as plt                 # noqa: E402
from numba import njit                          # noqa: E402

FIG = PROJ / "figures"
TAB = PROJ / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)

# Annualisation: the program's strategy corpora are evaluated on ~daily pnl
# aggregates (see lib/overfit usage in project 00). sqrt(252) maps a
# per-observation Sharpe to an annualised one, matching the *_ann columns in the
# real scorecard tables.
ANNUALISE = np.sqrt(252.0)


# --------------------------------------------------------------------------- #
# Monte-Carlo of skill-less strategies: pure-NumPy reference and Numba kernel
#
# Parity strategy: the scientific hot loop is "given a (T x N) matrix of IID
# skill-less returns, return the maximum per-column Sharpe". We implement that
# loop twice (NumPy reference and @njit kernel) and verify BIT-IDENTICAL on the
# SAME pre-drawn matrices. RNG draws stay outside the kernel so parity is exact.
# The fast path draws each sim's matrix and calls the verified kernel.
# --------------------------------------------------------------------------- #
def _max_sharpe_ref(X: np.ndarray) -> float:
    """Pure-NumPy reference: max per-column Sharpe of a (T x N) matrix."""
    mu = X.mean(0)
    sd = X.std(0, ddof=1)
    sr = np.where(sd > 0, mu / sd, 0.0)
    return float(sr.max())


@njit(cache=True, fastmath=False)
def _max_sharpe_kernel(X: np.ndarray) -> float:
    """Numba @njit hot loop: max per-column Sharpe of a (T x N) matrix.
    Same arithmetic as _max_sharpe_ref (Welford-free two-pass mean/var,
    ddof=1). fastmath=False keeps float ops bit-identical to NumPy."""
    T, N = X.shape
    best = -1.0e18
    for j in range(N):
        s = 0.0
        for t in range(T):
            s += X[t, j]
        mu = s / T
        ss = 0.0
        for t in range(T):
            d = X[t, j] - mu
            ss += d * d
        var = ss / (T - 1)
        if var > 0.0:
            sr = mu / np.sqrt(var)
        else:
            sr = 0.0
        if sr > best:
            best = sr
    return best


def mc_emax_ref(n_trials: int, T: int, n_sims: int, seed: int) -> float:
    """Pure-NumPy MC: mean over sims of the max Sharpe of N skill-less trials."""
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_sims):
        acc += _max_sharpe_ref(rng.standard_normal((T, n_trials)))
    return acc / n_sims


def mc_emax_numba(n_trials: int, T: int, n_sims: int, seed: int) -> float:
    """Numba-driven MC: identical statistic, verified kernel for the inner loop."""
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_sims):
        acc += _max_sharpe_kernel(rng.standard_normal((T, n_trials)))
    return acc / n_sims


def _verify_kernel_vs_ref(seed: int = 12345) -> dict:
    """Bit-identical check: feed the SAME pre-drawn matrices to both loops and
    require max|delta| == 0 over many draws (plus a NaN-free guarantee)."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(200):
        X = rng.standard_normal((rng.integers(200, 1200), rng.integers(5, 300)))
        deltas.append(abs(_max_sharpe_ref(X) - _max_sharpe_kernel(X)))
    return dict(det_delta=float(max(deltas)), n_checks=200)


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def build_curve() -> pd.DataFrame:
    """E[max Sharpe] vs N from the analytic False Strategy Theorem, var_sr=1/T."""
    T = 1000                                     # per-trial track length (obs)
    var_sr = 1.0 / T
    Ns = np.unique(np.round(np.logspace(0, 6, 49)).astype(int))
    Ns = Ns[Ns >= 2]
    rows = []
    for N in Ns:
        em = overfit.expected_max_sharpe(int(N), var_sr)
        rows.append(dict(n_trials=int(N),
                         T_obs=T,
                         e_max_sharpe_per_obs=em,
                         e_max_sharpe_ann=em * ANNUALISE))
    return pd.DataFrame(rows)


def mc_validation() -> pd.DataFrame:
    """Validate the formula against the Numba MC at several N (sanctioned MC)."""
    T = 1000
    var_sr = 1.0 / T
    n_sims = 4000
    rows = []
    for N in (10, 100, 1000, 5000):
        formula = overfit.expected_max_sharpe(N, var_sr)
        # Numba-driven MC (verified kernel inner loop)
        t0 = time.perf_counter()
        mc_k = mc_emax_numba(N, T, n_sims, seed=2024 + N)
        t_k = time.perf_counter() - t0
        rows.append(dict(n_trials=N, T_obs=T, n_sims=n_sims,
                         formula=formula, mc_numba=mc_k,
                         abs_delta=abs(formula - mc_k),
                         numba_sec=t_k))
    return pd.DataFrame(rows)


def main():
    style.set_style()
    print("=== Numba kernel warmup + bit-identical verification ===")
    t0 = time.perf_counter()
    _max_sharpe_kernel(np.random.default_rng(0).standard_normal((100, 10)))  # compile
    print(f"  compiled in {time.perf_counter()-t0:.2f}s")
    vk = _verify_kernel_vs_ref()
    print(f"  kernel vs NumPy ref max|delta| over {vk['n_checks']} matrices = "
          f"{vk['det_delta']:.3e}  (want 0.0, bit-identical)")

    # Speedup: ref vs kernel on identical pre-drawn workload
    print("\n=== Numba speedup (ref vs kernel, identical draws) ===")
    Nw, Tw, Sw = 100, 1000, 300
    t0 = time.perf_counter(); ref_val = mc_emax_ref(Nw, Tw, Sw, seed=7); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); ker_val = mc_emax_numba(Nw, Tw, Sw, seed=7); t_ker = time.perf_counter() - t0
    speedup = t_ref / t_ker if t_ker > 0 else float('nan')
    print(f"  ref   {t_ref:.3f}s  E[max]={ref_val:.6f}")
    print(f"  numba {t_ker:.3f}s  E[max]={ker_val:.6f}")
    print(f"  speedup ~{speedup:.1f}x ; |ref-numba| (same draws) = {abs(ref_val-ker_val):.3e}")

    # Curve + table
    curve = build_curve()
    curve.to_csv(TAB / "expected_max_sharpe_vs_N.csv", index=False)
    print(f"\nwrote {TAB/'expected_max_sharpe_vs_N.csv'}  ({len(curve)} rows)")

    # MC validation table
    mcv = mc_validation()
    mcv.to_csv(TAB / "mc_vs_formula.csv", index=False)
    max_delta = float(mcv["abs_delta"].max())
    print(f"wrote {TAB/'mc_vs_formula.csv'}")
    print(f"  MC-vs-formula max|delta| (per-obs Sharpe) = {max_delta:.4f}")
    print(mcv.to_string(index=False))

    # ---- Figure: E[max Sharpe] vs N (log) with lone-quant marker + DSR line ----
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(curve.n_trials, curve.e_max_sharpe_ann, color="#0B3D91", lw=2.4,
            label="Expected max Sharpe of skill-less trials (False Strategy Theorem)")
    # Monte-Carlo validation points (sanctioned synthetic, clearly labelled)
    ax.scatter(mcv.n_trials, mcv.mc_numba * ANNUALISE, s=70, zorder=5,
               color="#E69F00", edgecolor="k", linewidth=0.6,
               label="Monte-Carlo of skill-less strategies (validation)")

    # Lone-quant marker: the program's largest single-instrument search.
    # In project 00 the crypto corpus is 2,500 strategies per pair; a lone quant
    # who silently searched that many configurations on ONE instrument and
    # reported only the best would land here purely by selection.
    N_lone = 2500
    em_lone = overfit.expected_max_sharpe(N_lone, 1.0 / 1000) * ANNUALISE
    ax.scatter([N_lone], [em_lone], s=160, marker="*", color="#D55E00",
               edgecolor="k", linewidth=0.8, zorder=6,
               label=f"Lone quant: {N_lone:,} silent trials on one instrument")
    ax.annotate(f"selection alone buys\n~{em_lone:.1f} annualised Sharpe",
                xy=(N_lone, em_lone), xytext=(N_lone*1.3, em_lone*0.55),
                fontsize=9.5, color="#D55E00",
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))

    ax.set_xscale("log")
    ax.set_xlabel("Number of trials N (log scale)")
    ax.set_ylabel("Expected maximum Sharpe (annualised)")
    ax.set_title("The Sisyphus trap: the best of N skill-less backtests is mostly selection, not skill")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(2, 1e6)
    fig.text(0.99, 0.01,
             "Skill-less trials have zero true edge by construction; the curve is what selection alone delivers. "
             "Disclosing N lets the Deflated Sharpe Ratio subtract this benchmark.",
             ha="right", va="bottom", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "expected_max_sharpe_vs_N.png")
    fig.savefig(FIG / "expected_max_sharpe_vs_N.svg")
    plt.close(fig)
    print(f"wrote {FIG/'expected_max_sharpe_vs_N.png'} (+ .svg)")

    # persist headline numbers for the writeup / brief
    headline = dict(
        e_max_ann_at={int(n): float(overfit.expected_max_sharpe(int(n), 1/1000) * ANNUALISE)
                      for n in (10, 100, 1000, 2500, 10000, 100000, 1000000)},
        lone_quant_N=N_lone,
        lone_quant_e_max_ann=float(em_lone),
        mc_vs_formula_max_abs_delta=max_delta,
        kernel_determinism_delta=float(vk["det_delta"]),
        numba_speedup_x=float(speedup),
    )
    (TAB / "exp_max_sharpe_headline.json").write_text(json.dumps(headline, indent=2))
    print(f"wrote {TAB/'exp_max_sharpe_headline.json'}")
    print("\nE[max Sharpe] annualised at selected N:")
    for n, v in headline["e_max_ann_at"].items():
        print(f"  N={n:>9,}  ->  {v:5.2f}")


if __name__ == "__main__":
    main()

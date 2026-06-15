#!/usr/bin/env python3
"""Figure for the real-data positive control: detection probability versus the planted
edge's net annualised Sharpe, for the single-hypothesis benchmark and the corpus-scale
benchmark. Regenerates the calibration figure directly from positive_control_real.py.

Usage:
    python3 positive_control/make_fig_poscontrol.py [output.pdf]
The output path defaults to fig_poscontrol.pdf next to this script. The BTCUSDT 30-minute
root is read from the LDP_CRYPTO_30M environment variable (see ../config.py, ../DATA.md).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import positive_control_real as PC

SINGLE_FLOOR = 0.7
CORPUS_FLOOR = 3.4
BEST_OBSERVED = PC.BEST_REAL_CORPUS_SHARPE   # 2.0
EMAX = PC.SR0_CORPUS_ANN                      # 2.72


def detection_curves():
    r = PC.load_btc_returns()
    T = len(r)
    sigma = float(r.std(ddof=1))
    sr0_corpus = EMAX / PC.ANN
    alphas = np.round(np.arange(0.0, 0.046, 0.0015), 5)
    net_sr, det_single, det_corpus = [], [], []
    for a in alphas:
        srs, ps, pc = [], [], []
        for seed in range(PC.N_REAL):
            pnl = PC.strat_returns(r, PC.make_position(T, seed), a, sigma)
            ok_s, nsr = PC.passes_bar(pnl, 0.0)
            ok_c, _ = PC.passes_bar(pnl, sr0_corpus)
            srs.append(nsr); ps.append(ok_s); pc.append(ok_c)
        net_sr.append(np.mean(srs)); det_single.append(np.mean(ps)); det_corpus.append(np.mean(pc))
    return np.array(net_sr), np.array(det_single), np.array(det_corpus)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_poscontrol.pdf")
    x, single, corpus = detection_curves()

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(x, single, "-o", color="#27ae60", ms=4, lw=2,
            label="single hypothesis (deflation vs 0)")
    ax.plot(x, corpus, "--s", color="#2980b9", ms=4, lw=2,
            label=r"corpus search size (deflation vs $E[\max]=2.72$)")
    ax.axvline(SINGLE_FLOOR, color="#27ae60", ls=":", lw=1.2)
    ax.axvline(CORPUS_FLOOR, color="#2980b9", ls=":", lw=1.2)
    ax.axvline(EMAX, color="#c0392b", ls="-", lw=1.4)
    ax.axvline(BEST_OBSERVED, color="0.35", ls=":", lw=1.2)
    # labels nudged just right of their lines and drawn above them (zorder) so the
    # dotted vline never clips the first character; legend moved below the axes so it
    # no longer overlaps the vertical labels.
    tkw = dict(fontsize=8, rotation=90, ha="left", zorder=6)
    ax.text(SINGLE_FLOOR + 0.04, 0.50, r"floor $\approx0.7$", color="#1e8449", va="center", **tkw)
    ax.text(BEST_OBSERVED + 0.04, 0.50, "best observed 2.0", color="0.35", va="center", **tkw)
    ax.text(CORPUS_FLOOR + 0.04, 0.34, r"floor $\approx3.4$", color="#1f618d", va="center", **tkw)
    ax.text(EMAX + 0.04, 0.05, r"$E[\max]=2.72$", color="#c0392b", va="bottom", **tkw)

    ax.set_xlabel("Planted edge: net annualized Sharpe (after real costs)")
    ax.set_ylabel("Detection probability")
    ax.set_title("The bar accepts a true edge above a calibrated floor")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.2, x.max() + 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

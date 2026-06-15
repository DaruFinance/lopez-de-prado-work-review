#!/usr/bin/env python3
"""Reproduce the apex order-flow ablation vector and figure from two saved WFO runs.

Reproduction (two arms of the committed pipeline; see run_clean_audit.py for the run):

  # 1. baseline (all features, both defects fixed)  -> runs/XS_clean_lgbm/summary.csv
  python run_clean_audit.py

  # 2. ablated arm: identical pipeline, order-flow block dropped
  XS_ABLATE=of_buy,of_delta XS_OUT_ROOT=runs_ablated \
      python -c "import xs_run as xr; xr.run('XS_ablate_of', \
                 xr.build_combos(families=['lgbm']), out_root='runs_ablated', nproc=4)"

  # 3. assemble the vector + regenerate the figure
  BASELINE_SUMMARY=runs/XS_clean_lgbm/summary.csv \
  ABLATED_SUMMARY=runs_ablated/XS_ablate_of/summary.csv \
      python reproduce_ablation.py [out_fig.pdf]

Dropping of_buy+of_delta (the 4h/8h forward-filled order-flow block) collapses the profit
factor to break-even at every horizon — the economic magnitude was the leak.
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.environ.get("BASELINE_SUMMARY", os.path.join(HERE, "runs/XS_clean_lgbm/summary.csv"))
ABLATED = os.environ.get("ABLATED_SUMMARY", os.path.join(HERE, "runs_ablated/XS_ablate_of/summary.csv"))
TABLES = os.path.join(HERE, "tables")
RED = "#c0392b"; BLUE = "#2c7fb8"; H_SHORT = 4


def load(path):
    with open(path) as f:
        return {int(r["H"]): r for r in csv.DictReader(f)}


def main(out_fig):
    base, abl = load(BASELINE), load(ABLATED)
    os.makedirs(TABLES, exist_ok=True)
    rows = []
    for H in sorted(base):
        if H not in abl:
            continue
        pf_b, pf_a = float(base[H]["oos_pf"]), float(abl[H]["oos_pf"])
        rows.append(dict(H=H, pf_with_orderflow=round(pf_b, 4), pf_ablated=round(pf_a, 4),
                         pf_drop=round(pf_b - pf_a, 4),
                         sharpe_with_orderflow=round(float(base[H]["oos_sharpe"]), 4),
                         sharpe_ablated=round(float(abl[H]["oos_sharpe"]), 4)))
    with open(os.path.join(TABLES, "orderflow_ablation_by_horizon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    pf_b = next(r["pf_with_orderflow"] for r in rows if r["H"] == H_SHORT)
    pf_a = next(r["pf_ablated"] for r in rows if r["H"] == H_SHORT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar([0, 1], [pf_b, pf_a], width=0.6, color=[RED, BLUE])
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(1.46, 1.0, "break-even", ha="right", va="bottom", fontsize=9)
    for x, v in zip([0, 1], [pf_b, pf_a]):
        ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=12, xytext=(0, 2), textcoords="offset points")
    ax.annotate("entire edge =\norder-flow leak", xy=(1, pf_a + 0.06), xytext=(0.5, (pf_b + pf_a) / 2),
                ha="center", va="center", fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#777777", connectionstyle="arc3,rad=-0.3"))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Reported\n(with feature)", "Leak-ablated\n(feature removed)"])
    ax.set_ylabel(f"Profit factor ({H_SHORT}-bar horizon)")
    ax.set_ylim(0, max(pf_b, 1.2) * 1.18)
    ax.set_title("Ablating one leaked feature collapses the survivor")
    fig.savefig(out_fig); plt.close(fig)
    print(f"vector -> tables/orderflow_ablation_by_horizon.csv ; figure -> {out_fig}")
    print(f"H{H_SHORT}: with order-flow {pf_b:.3f} -> ablated {pf_a:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_apex.pdf"))

"""
assembly_line_diagram.py: the meta-strategy assembly-line schematic.

Draws Lopez de Prado's "production line" of specialised, separable stations and
maps each completed study in this program to the station(s) it stress-tested.
Pure matplotlib (no external data); the station->study mapping mirrors the real
scorecard produced by meta_analysis.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT))
from lib import style                            # noqa: E402

FIG = PROJ / "figures"
FIG.mkdir(exist_ok=True)

# Station name, the role, and the studies that stress-tested it.
STATIONS = [
    ("Data\ncurators", "raw data -> clean,\ncausal bars",
     ["01 Information-driven bars", "02 Fractional differentiation"]),
    ("Feature\nanalysts", "informative,\nnon-leaking features",
     ["07 Structural breaks & entropy", "08 Microstructural features", "14 Causal factors"]),
    ("Strategists", "labels + sides\n(the bet)",
     ["03 Triple-barrier + meta-label", "05 Trend-scanning labels"]),
    ("Backtesters /\nvalidation", "purged CV, DSR,\nPBO gate",
     ["04 Purged CV / CPCV", "09 Ensembles & importance", "00 Overfitting & DSR"]),
    ("Deployment /\nsizing", "bet sizing,\nexecution rules",
     ["11 Bet sizing", "12 Optimal trading rules"]),
    ("Portfolio\noversight", "allocation across\nstrategies",
     ["13 Portfolio construction"]),
]

COLORS = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7", "#0072B2"]


def main():
    style.set_style()
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    ax.set_xlim(0, len(STATIONS) * 2.35)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    box_w, box_h = 1.95, 1.4
    y_box = 6.0
    centers = []
    for i, (name, role, studies) in enumerate(STATIONS):
        x = i * 2.35 + 0.2
        cx = x + box_w / 2
        centers.append(cx)
        # station box
        ax.add_patch(FancyBboxPatch((x, y_box), box_w, box_h,
                     boxstyle="round,pad=0.04,rounding_size=0.12",
                     linewidth=1.6, edgecolor=COLORS[i], facecolor=COLORS[i] + "22"))
        ax.text(cx, y_box + box_h * 0.66, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#222222")
        ax.text(cx, y_box + box_h * 0.24, role, ha="center", va="center",
                fontsize=8.2, color="#444444", style="italic")
        # studies that stress-test this station
        for k, s in enumerate(studies):
            yy = y_box - 0.55 - k * 0.62
            ax.add_patch(FancyBboxPatch((x + 0.05, yy - 0.24), box_w - 0.1, 0.46,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=0.9, edgecolor=COLORS[i], facecolor="white"))
            ax.text(cx, yy, s, ha="center", va="center", fontsize=7.0, color="#333333")

    # arrows between stations (the conveyor)
    for i in range(len(STATIONS) - 1):
        a = FancyArrowPatch((centers[i] + box_w / 2 - 0.05, y_box + box_h / 2),
                            (centers[i + 1] - box_w / 2 + 0.05, y_box + box_h / 2),
                            arrowstyle="-|>", mutation_scale=18,
                            linewidth=2.0, color="#888888")
        ax.add_patch(a)

    # feedback loop (validation gates everything; disclosure of N flows back),
    # drawn well below the study boxes so nothing overlaps.
    y_fb = 1.75
    ax.annotate("", xy=(centers[0], y_fb), xytext=(centers[3], y_fb),
                arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.8,
                                connectionstyle="arc3,rad=-0.18", linestyle="--"))
    ax.text((centers[0] + centers[3]) / 2, 0.92,
            "mandatory disclosure: every trial from every station is logged so the validation\n"
            "station can deflate the reported best by the true number of trials N",
            ha="center", va="center", fontsize=9, color="#D55E00")

    ax.text(ax.get_xlim()[1] / 2, 8.25,
            "The meta-strategy assembly line: specialised, separable stations with one shared validation gate",
            ha="center", va="center", fontsize=13.5, fontweight="bold")
    ax.text(ax.get_xlim()[1] / 2, 0.18,
            "Each station is a separable role, not a lone quant doing everything. The opposite, the Sisyphus "
            "quant who finds strategies by repeatedly backtesting one idea, hides N and overfits by construction.",
            ha="center", va="center", fontsize=8.6, color="#555555")

    fig.tight_layout()
    fig.savefig(FIG / "assembly_line.png")
    fig.savefig(FIG / "assembly_line.svg")
    plt.close(fig)
    print(f"wrote {FIG/'assembly_line.png'} (+ .svg)")


if __name__ == "__main__":
    main()

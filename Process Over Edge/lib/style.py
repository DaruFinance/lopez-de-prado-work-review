"""style.py, shared, clean matplotlib defaults for publication-quality figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# colour-blind-safe palette
PALETTE = {
    "time":   "#999999",   # grey , the control
    "tick":   "#56B4E9",   # blue
    "volume": "#E69F00",   # orange
    "dollar": "#009E73",   # green, LdP's preferred bar
    "accent": "#D55E00",
}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def barcolor(name: str) -> str:
    return PALETTE.get(name, "#333333")

#!/usr/bin/env python3
"""
Value-add, does the minimum d* depend on the sampling frequency of the base
series?  Compares FFD d* on 1-minute vs 1-hour log-price for the five pairs we
have at 1m granularity (BTC, ETH, SOL, DOGE, BNB).

LdP works at one frequency; a natural question for a trading pipeline is whether
the fractional order is a property of the *price process* (frequency-invariant)
or of the *sampling*.  Causal FFD only; run: python3 run_frequency_study.py
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.exists(_os.path.join(_d, "config.py")):
    _d = _os.path.dirname(_d)
REPO_ROOT = _d
_sys.path.insert(0, REPO_ROOT)
import config as cfg
from config import LIB as _LIB
_sys.path.insert(0, _LIB)
import fracdiff as F
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(PROJ, "figures"), os.path.join(PROJ, "tables")
ONEM = cfg.CRYPTO_1M
ONEH = cfg.CRYPTO_1H
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]
TAU = 1e-5
D_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)


def logclose(path):
    c = pd.to_numeric(pd.read_parquet(path, columns=["close"])["close"], errors="coerce").to_numpy()
    c = c[np.isfinite(c)]
    c = c[c > 0]
    return np.log(c)


def main():
    rows = []
    for p in PAIRS:
        for label, path in [("1m", f"{ONEM}/{p}_1m.parquet"),
                            ("1h", f"{ONEH}/{p}_1h.parquet")]:
            if not os.path.exists(path):
                continue
            x = logclose(path)
            res = F.min_d_search(x, d_grid=D_GRID, tau=TAU)
            s = res["star"]
            rows.append(dict(pair=p, base=label, n=len(x),
                             d_star=res["d_star"],
                             corr_level_at_dstar=(s["corr_level"] if s else np.nan),
                             window_len=(s["window_len"] if s else np.nan)))
            print(p, label, "d* =", res["d_star"], "n =", len(x))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "dstar_1m_vs_1h.csv"), index=False)

    piv = df.pivot(index="pair", columns="base", values="d_star").reindex(PAIRS)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xpos = np.arange(len(piv))
    w = 0.36
    ax.bar(xpos - w / 2, piv.get("1m"), w, color=ST.PALETTE["tick"], label="1m base")
    ax.bar(xpos + w / 2, piv.get("1h"), w, color=ST.PALETTE["dollar"], label="1h base")
    ax.set_xticks(xpos)
    ax.set_xticklabels(piv.index, rotation=20)
    ax.set_ylabel("minimum d* for stationarity")
    ax.set_title("d* by base sampling frequency (FFD, tau=1e-5)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig8_dstar_1m_vs_1h.png"))
    plt.close(fig)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

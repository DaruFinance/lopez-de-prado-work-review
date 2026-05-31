#!/usr/bin/env python3
"""
Project 1 (forex) — Information bars on spot FX, where the only information clock
is TICK COUNT (spot FX has no real volume). We compare TIME bars vs TICK bars
across 8 majors from HistData quote ticks (1-min base with tick-count).

FX trades ~24x5, so returns spanning the weekend / daily rollover gap are dropped
(gap-aware), mirroring the session handling that mattered for equities.

Outputs: tables/forex_bars.csv, tables/forex_bars.md, figures/fig_fx_kurtosis.png
"""
import sys, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/daru/ldp_review/lib")
import bars as B
import barstats as S
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = "/home/daru/ldp_review/projects/01_information_driven_bars"
CACHE = "/mnt/c/Users/USUARIO/Desktop/ldp_cache_fx"
GAP_HOURS = 2.0            # drop bar returns spanning a market-closed gap


def load_fx(path):
    df = pd.read_parquet(path)
    # add proxy columns so the shared aggregator works; only count is real
    df["volume"] = df["count"].astype(float)
    df["quote_volume"] = df["count"].astype(float)
    df["taker_buy_volume"] = df["count"] / 2.0
    df["taker_buy_quote_volume"] = df["count"] / 2.0
    return df[(df["close"] > 0) & (df["count"] > 0)]


def gap_filtered_returns(bars):
    """Log returns, dropping bars whose open follows the prior close by > GAP_HOURS."""
    lp = np.log(bars["close"].to_numpy())
    r = np.diff(lp)
    to_ns = pd.DatetimeIndex(pd.to_datetime(bars["t_open"], utc=True)).asi8
    tc_ns = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True)).asi8
    gap_h = (to_ns[1:] - tc_ns[:-1]) / 3.6e12          # ns -> hours
    r = r[gap_h <= GAP_HOURS]
    return pd.Series(r).dropna()


def main():
    files = sorted(glob.glob(f"{CACHE}/*_fx1m.parquet"))
    print(f"{len(files)} FX pairs")
    rec = []
    for f in files:
        name = os.path.basename(f).replace("_fx1m.parquet", "")
        df = load_fx(f)
        if len(df) < 30000:
            print("  short", name, len(df)); continue
        days = max(50, int((df.index[-1] - df.index[0]).days))
        bpb = max(1, round(len(df) / days))
        tbars = B.time_bars(df, bpb)
        kbars = B.threshold_bars(df, "count", df["count"].sum() / days)
        st_t = S.return_stats(gap_filtered_returns(tbars))
        st_k = S.return_stats(gap_filtered_returns(kbars))
        rec.append(dict(pair=name, exkurt_time=st_t["exkurt"], exkurt_tick=st_k["exkurt"],
                        absskew_time=abs(st_t["skew"]), absskew_tick=abs(st_k["skew"]),
                        ac1_time=abs(st_t["ac1"]), ac1_tick=abs(st_k["ac1"]),
                        n_time=st_t["n"], n_tick=st_k["n"]))
        print(f"  {name}: time exkurt {st_t['exkurt']:.2f} -> tick {st_k['exkurt']:.2f}")

    df = pd.DataFrame(rec)
    df.to_csv(f"{PROJ}/tables/forex_bars.csv", index=False)
    med = df[["exkurt_time", "exkurt_tick", "absskew_time", "absskew_tick"]].median()
    with open(f"{PROJ}/tables/forex_bars.md", "w") as fh:
        fh.write(f"# Forex: time vs tick bars ({len(df)} majors, median)\n\n")
        fh.write("Spot FX has no volume; tick count is the information clock. "
                 "Gap-aware returns (weekend/rollover dropped). Lower exkurt is better.\n\n")
        fh.write(df.round(3).to_markdown(index=False))
        fh.write("\n\n**Medians:** time exkurt %.2f vs tick exkurt %.2f; "
                 "|skew| %.3f vs %.3f\n" % (med.exkurt_time, med.exkurt_tick,
                                            med.absskew_time, med.absskew_tick))
    print("\n=== FOREX medians ===")
    print(med.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(df)); w = 0.4
    ax.bar(x - w/2, df["exkurt_time"], w, label="time bars", color=ST.barcolor("time"))
    ax.bar(x + w/2, df["exkurt_tick"], w, label="tick bars", color=ST.barcolor("tick"))
    ax.set_xticks(x); ax.set_xticklabels(df["pair"], rotation=45, ha="right")
    ax.set_ylabel("Excess kurtosis of bar returns")
    ax.set_title("Forex: tick bars Gaussianize returns vs time bars\n"
                 "(spot FX has no volume → tick count is the information clock)")
    ax.legend()
    fig.savefig(f"{PROJ}/figures/fig_fx_kurtosis.png"); plt.close(fig)
    print("\nForex bars study done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""run_us_equity_xsection.py : US-EQUITY cross-sectional tree rotation, built to
the SAME recipe as the crypto cross-sectional study (the canonical Gu/Kelly/Xiu
setting): per bar, rank a bounded liquid universe by a CAUSAL per-name tree score,
rotate long the top quantile / short the bottom quantile (market-neutral), tune the
rotation knobs in-sample per purged WFO window, charge realistic per-fill costs,
then score OOS profit-factor / Sharpe / Deflated Sharpe per horizon and compare to
the crypto cross-sectional result.

It reuses, UNMODIFIED, the program's cross-sectional engine module
(ml_xsection_corpus.py): build_features, forward_rank_label, make_model,
simulate, wfo_windows, run_combo. That engine is produced by a separate pipeline
and is not bundled here; point config.XS_ENGINE (env LDP_XS_ENGINE) at the
directory holding it. The ONLY thing this script adds is a panel loader for the
downloaded adjusted daily US-equity OHLC (the engine's own equity path is a
9-ETF rotation; here we feed it a several-hundred-name large-cap cross-section),
plus the per-horizon DSR table, the figure, and the crypto compare.

Single process. RAM-bounded: the panel is a (T x N) float64 close matrix
(~2900 x 244 ~= 5.7 MB) plus a (T x N x feats) feature cube (~hundreds of MB); we
hold ONE copy and run combos sequentially. No look-ahead: features are shifted one
bar inside the engine; the label is a forward H-day cross-sectional return rank;
positions decided at t earn the t->t+1 return.

All data and engine roots come from the repo config (config.py); no paths are
hard-coded here.
"""
from __future__ import annotations
import os, sys, glob, gzip, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats as ss

warnings.filterwarnings("ignore")

_d = os.path.dirname(os.path.abspath(__file__))
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d                                  # repo root (holds config.py)
sys.path.insert(0, ROOT)
import config as cfg                       # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, cfg.XS_ENGINE)         # dir holding ml_xsection_corpus.py

import overfit as O                       # noqa: E402  the program's DSR harness
from lib import realism as RZ             # noqa: E402  causal per-fill costs
import ml_xsection_corpus as XS           # noqa: E402  REUSED ENGINE (unmodified)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
TAB = os.path.join(STUDY, "tables")
FIG = os.path.join(STUDY, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DATA_DIR = os.environ.get("EQ_XS_DIR", cfg.EQUITY_DAILY_XS)

# Daily horizons (trading days): the canonical equity cross-section set
# {1d, 1w, 2w, 1m, 1q}, the analogue of the crypto study's horizon family.
HORIZONS = [int(h) for h in os.environ.get("EQ_HORIZONS", "1,5,10,21,63").split(",")]
FAMILY = os.environ.get("EQ_FAMILY", "lgbm")          # lgbm (catboost optional)
MIN_NAMES_PER_BAR = 20                                 # require a real cross-section

# Crypto cross-sectional banked headline (787-pair panel, lgbm), for the compare.
CRYPTO_BANKED = dict(
    family="lgbm", n_names="787 perp pairs", median_oos_pf=1.123,
    median_oos_sharpe_ann=2.66, dsr_survive="10/10 horizons", pbo=0.10,
    source="crypto cross-sectional rotation (XS_main); program headline",
)


def _note(m): print(f"  [note] {m}", flush=True)
def _ok(m):   print(f"  [ok]   {m}", flush=True)


# --------------------------------------------------------------------------- #
# Panel: adjusted daily OHLC for the downloaded large-cap universe.
# Same shape the engine expects: close/ret/valid/per_side/symbols/grid/T/N.
# --------------------------------------------------------------------------- #
def load_equity_panel() -> dict:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv.gz")))
    if not files:
        raise RuntimeError(f"no daily files in {DATA_DIR} - run fetch_us_equity_daily.py first")
    closes, highs, lows, syms = {}, {}, {}, []
    for fp in files:
        df = pd.read_csv(fp, usecols=["TradeDate", "Ticker", "OpenPrice", "HighPrice",
                                      "LowPrice", "ClosePrice", "DailyVolume"])
        if df.empty:
            continue
        tk = str(df["Ticker"].iloc[0])
        df = df.assign(d=pd.to_datetime(df["TradeDate"])).set_index("d").sort_index()
        df = df[df["ClosePrice"] > 0]
        if len(df) < (XS.IS + XS.WALK):
            continue
        closes[tk] = df["ClosePrice"]; highs[tk] = df["HighPrice"]; lows[tk] = df["LowPrice"]
        syms.append(tk)
    if len(syms) < 4:
        raise RuntimeError(f"only {len(syms)} usable names (<4) - cannot cross-section")
    close = pd.concat([closes[s] for s in syms], axis=1, keys=syms).sort_index()
    grid = pd.DatetimeIndex(close.index)
    C = close.to_numpy(np.float64)
    T, N = C.shape
    valid = np.isfinite(C) & (C > 0)
    # within-live-span gate: no look-ahead onto pre-IPO leading NaNs (later IPOs
    # like ABNB/COIN simply enter the cross-section once they list).
    for j in range(N):
        idxs = np.where(valid[:, j])[0]
        if len(idxs):
            valid[:idxs[0], j] = False
            valid[idxs[-1] + 1:, j] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(C)
    ret = np.zeros((T, N)); ret[1:] = lp[1:] - lp[:-1]
    ret[~np.isfinite(ret)] = 0.0
    # realistic per-side equity cost (time-of-day half-spread + commission floor),
    # evaluated at the RTH midday stamp; causal (time-of-day is ex-ante).
    per_side = np.zeros(N)
    rep = grid.tz_localize("America/New_York") + pd.Timedelta(hours=12)
    for j, s in enumerate(syms):
        cj = C[:, j]
        med = np.nanmedian(cj[valid[:, j]]) if valid[:, j].any() else np.nan
        px = np.where(cj > 0, cj, med)
        per_side[j] = float(np.nanmedian(RZ.per_side_cost_fraction("equities", s, rep, px)))
    per_side = np.nan_to_num(per_side, nan=float(np.nanmedian(per_side)))
    return dict(close=C, ret=ret, valid=valid, per_side=per_side,
                symbols=np.array(syms), grid=grid, T=T, N=N)


# --------------------------------------------------------------------------- #
# Per-horizon OOS PF / Sharpe / DSR. DSR deflates each horizon against the
# dispersion of the horizon-family OOS Sharpes (the trials searched).
# --------------------------------------------------------------------------- #
def per_horizon_table(per_h_oos: dict) -> pd.DataFrame:
    Hs = sorted(per_h_oos)
    sr_trials = np.array([O.sharpe(per_h_oos[H]) for H in Hs])
    rows = []
    for H in Hs:
        oos = np.asarray(per_h_oos[H], float)
        nz = oos[oos != 0.0]
        pos = nz[nz > 0].sum(); neg = -nz[nz < 0].sum()
        pf = float(pos / neg) if neg > 0 else np.nan
        sr = O.sharpe(oos)
        sk = float(ss.skew(oos)) if len(oos) > 2 else 0.0
        ku = float(ss.kurtosis(oos, fisher=False)) if len(oos) > 2 else 3.0
        d = O.deflated_sharpe_ratio(sr, len(oos), sk, ku, sr_trials)
        rows.append(dict(
            H=H, oos_bars=len(oos), oos_pf=round(pf, 4),
            oos_sharpe=round(sr, 4), oos_sharpe_ann=round(sr * np.sqrt(252), 3),
            skew=round(sk, 3), kurt=round(ku, 3),
            sr0_benchmark=round(d["sr0"], 4), dsr=round(d["dsr"], 4),
            dsr_survives=bool(d["dsr"] > 0.95),
        ))
    return pd.DataFrame(rows).sort_values("H").reset_index(drop=True)


def make_figure(tab: pd.DataFrame):
    try:
        import style as STY
        STY.set_style()
    except Exception as e:
        _note(f"style unavailable ({e}); using matplotlib defaults")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    H = tab["H"].astype(str)
    ax[0].bar(H, tab["oos_pf"], color="#0072B2")
    ax[0].axhline(1.0, ls="--", lw=0.8, color="#999999")
    ax[0].axhline(CRYPTO_BANKED["median_oos_pf"], ls=":", lw=1.0, color="#D55E00")
    ax[0].text(len(H) - 0.5, CRYPTO_BANKED["median_oos_pf"] + 0.002,
               f"crypto median ~{CRYPTO_BANKED['median_oos_pf']:.3f}",
               color="#D55E00", fontsize=8, ha="right")
    ax[0].set_xlabel("horizon H (trading days)")
    ax[0].set_ylabel("OOS profit factor")
    ax[0].set_title("US-equity cross-sectional lgbm OOS PF by horizon")
    lo = min(0.97, float(tab["oos_pf"].min()) * 0.99)
    ax[0].set_ylim(lo, max(1.16, float(tab["oos_pf"].max()) * 1.04))

    ax[1].bar(H, tab["dsr"], color="#56B4E9")
    ax[1].axhline(0.95, ls="--", lw=0.8, color="#D55E00")
    ax[1].text(len(H) - 0.5, 0.96, "DSR 0.95", color="#D55E00", fontsize=8, ha="right")
    ax[1].set_xlabel("horizon H (trading days)")
    ax[1].set_ylabel("Deflated Sharpe Ratio")
    ax[1].set_title("US-equity cross-sectional lgbm DSR by horizon")
    ax[1].set_ylim(0, 1.05)
    fig.suptitle("Does the cross-sectional ML edge replicate in US equities?",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_us_equity_xsection.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    _ok("figures/fig_us_equity_xsection.png")


def main():
    print("US-EQUITY cross-sectional ML rotation - same recipe as the crypto study", flush=True)
    print("=" * 72, flush=True)
    panel = load_equity_panel()
    span = f"{panel['grid'][0].date()}..{panel['grid'][-1].date()}"
    per_bar_n = panel["valid"].sum(1)
    med_alive = int(np.median(per_bar_n[per_bar_n > 0]))
    _ok(f"panel: N={panel['N']} names | T={panel['T']} daily bars | {span} | "
        f"median alive/bar={med_alive} | per-side cost med={np.median(panel['per_side'])*1e4:.1f}bp")
    if med_alive < MIN_NAMES_PER_BAR:
        _note(f"median alive/bar {med_alive} < {MIN_NAMES_PER_BAR}: thin cross-section")

    X, names = XS.build_features(panel)
    _ok(f"features: {X.shape[2]} per-name causal features (rank+z of {names})")

    # One combo per horizon (family x H), exactly as the engine defines a strategy.
    per_h_oos, summ_rows = {}, []
    for cid, H in enumerate(HORIZONS):
        spec = dict(family=FAMILY, H=H, combo_id=cid)
        summ, recs = XS.run_combo(panel, X, spec)
        if summ is None:
            _note(f"H={H}: no OOS windows produced - skipped")
            continue
        # rebuild the concatenated OOS per-bar stream from the ledger records
        oos = np.array([v for (c, w, ph, b, v) in recs if ph == 1], float)
        per_h_oos[H] = oos
        summ_rows.append(summ)
        _ok(f"H={H:>3}d: OOS bars={len(oos):>5} | PF={summ['oos_pf']:.4f} | "
            f"Sharpe(ann)={summ['oos_sharpe_ann']:.3f}")

    if not per_h_oos:
        raise RuntimeError("no horizon produced OOS streams - check WFO geometry vs panel length")

    tab = per_horizon_table(per_h_oos)
    tab.to_csv(os.path.join(TAB, "us_equity_xsection.csv"), index=False)

    med_pf = float(tab["oos_pf"].median())
    med_sr_ann = float(tab["oos_sharpe_ann"].median())
    n_surv = int(tab["dsr_survives"].sum())
    replicates = (med_pf > 1.0) and (n_surv >= 1)

    # markdown table + crypto compare
    with open(os.path.join(TAB, "us_equity_xsection.md"), "w") as f:
        f.write("# US-equity cross-sectional ML rotation: per-horizon OOS and DSR\n\n")
        f.write(f"Universe: {panel['N']} large-cap US-listed names (predominantly "
                f"S&P 500 members), "
                f"adjusted daily bars {span}, median {med_alive} alive per bar. "
                f"Family: {FAMILY}. Market-neutral long-short rotation, purged WFO "
                f"(IS={XS.IS}/walk={XS.WALK}/sel={XS.IS_SEL} days), realistic per-fill "
                f"costs (median {np.median(panel['per_side'])*1e4:.1f} bp/side).\n\n")
        f.write(tab.to_markdown(index=False))
        f.write("\n\n## Multi-market comparison: crypto vs US equities\n\n")
        comp = pd.DataFrame([
            dict(market="crypto perps", universe=CRYPTO_BANKED["n_names"],
                 family=CRYPTO_BANKED["family"],
                 median_oos_pf=CRYPTO_BANKED["median_oos_pf"],
                 median_oos_sharpe_ann=CRYPTO_BANKED["median_oos_sharpe_ann"],
                 dsr_survive=CRYPTO_BANKED["dsr_survive"], pbo=CRYPTO_BANKED["pbo"]),
            dict(market="US equities", universe=f"{panel['N']} large-cap names",
                 family=FAMILY, median_oos_pf=round(med_pf, 4),
                 median_oos_sharpe_ann=round(med_sr_ann, 3),
                 dsr_survive=f"{n_surv}/{len(tab)} horizons", pbo=None),
        ])
        f.write(comp.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Survivorship caveat.** The universe is a fixed list of CURRENT "
                "large-cap members, so names that were large-cap earlier but have "
                "since been removed (delistings, takeovers, demotions) are absent. "
                "This is a mild upward bias on the long side; the result is "
                "survivorship-aware but not survivorship-free. Later IPOs enter the "
                "cross-section only once they list (within-live-span gating), so "
                "there is no look-ahead onto pre-listing dates.\n\n")
        verdict = ("REPLICATES" if replicates else "does NOT replicate")
        f.write(f"**Verdict.** The cross-sectional ML edge {verdict} in US equities: "
                f"median OOS PF {med_pf:.3f} (crypto {CRYPTO_BANKED['median_oos_pf']:.3f}), "
                f"DSR survival {n_surv}/{len(tab)} horizons.\n")

    comp.to_csv(os.path.join(TAB, "us_equity_xsection.csv".replace(".csv", "_vs_crypto.csv")),
                index=False)
    make_figure(tab)

    summary = dict(
        market="us_equity", data_source="Algoseek eq-daily-ohlc (adjusted)",
        n_names=int(panel["N"]), span=span, median_alive_per_bar=med_alive,
        family=FAMILY, horizons=HORIZONS,
        per_side_cost_bp_med=round(float(np.median(panel["per_side"]) * 1e4), 3),
        median_oos_pf=round(med_pf, 4), median_oos_sharpe_ann=round(med_sr_ann, 3),
        dsr_survivors=f"{n_surv}/{len(tab)}", per_horizon=tab.to_dict("records"),
        crypto_banked=CRYPTO_BANKED, replicates=bool(replicates),
    )
    with open(os.path.join(TAB, "us_equity_xsection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _ok("tables/us_equity_xsection.{csv,md} + us_equity_xsection_summary.json")

    print(f"\n  RESULT: US-equity cross-sectional {FAMILY} | median OOS PF={med_pf:.3f} | "
          f"median Sharpe(ann)={med_sr_ann:.3f} | DSR>0.95: {n_surv}/{len(tab)}", flush=True)
    print(f"  CRYPTO : median OOS PF={CRYPTO_BANKED['median_oos_pf']:.3f} | "
          f"DSR {CRYPTO_BANKED['dsr_survive']} | PBO {CRYPTO_BANKED['pbo']}", flush=True)
    print(f"  EDGE {'REPLICATES' if replicates else 'DOES NOT REPLICATE'} in US equities.",
          flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

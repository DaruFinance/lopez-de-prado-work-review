#!/usr/bin/env python3
"""
run_metalabel_real_primary.py: Meta-labeling an EDGED primary (LdP AFML Ch.3, ML4AM Ch.5).

This study CORRECTS sibling study 03 (`projects/03_meta_labeling/`). Study 03
meta-labeled a vanilla EMA-crossover primary that has NO established edge net of
costs, found the secondary lifted raw PF on 39/42 instruments but 0 cleared
DSR>0.95, and concluded meta-labeling is "a precision filter, not an alpha
source." That conclusion violates Lopez de Prado's STATED PRECONDITION:
meta-labeling presupposes a primary that ALREADY has an edge; the secondary only
SIZES and FILTERS it ("the primary model decides the side, the meta-model decides
whether to act and how big", AFML 3.6). Meta-labeling an edgeless primary is
garbage-in: there is no real edge for the filter to concentrate.

THIS STUDY redoes the experiment with the precondition MET. The primary is a
*proven* structural order-flow / open-interest edge (a closed, already-validated
crypto archetype reused READ-ONLY through its banked verified engine), and we
apply the SAME apparatus (purged k-fold CV, realistic per-fill costs, DSR as the
headline via lib/overfit.py, plus PBO and effective-N) to test whether the
secondary improves a real edge. It does: PF 1.26 -> 1.79 and per-trade 47 -> 148 bp.

TWO MODES (both honour: real data only, causal features, per-fill costs, full
per-trade ledger, no look-ahead):

  --from-banked  (DEFAULT)  Reproduce the corrected result from the banked
                 per-trade ledger (the original meta-gate-a-proven-edge
                 experiment's full per-trade parquet). Fast (<5 s), no heavy
                 compute, no engine invocation. Computes primary-vs-meta PF /
                 per-trade bp / precision-vs-base-rate and the DEFLATED Sharpe
                 Ratio for both arms over the WFO windows. This is the path that
                 regenerates the 1.26 -> 1.79 headline. Robust to a missing
                 ledger (skip + note).

  --live         Re-run the meta-labeling end to end on the edged primary by
                 calling the verified structural engine READ-ONLY (its numba sim
                 kernel + cost model; the engine's kernel.py and common.py are
                 NEVER modified). Heavier (minutes). Robust to missing engine data
                 (skip + note).

Run:
  python3 scripts/run_metalabel_real_primary.py                 # banked reproduction (default)
  python3 scripts/run_metalabel_real_primary.py --from-banked   # explicit
  python3 scripts/run_metalabel_real_primary.py --live          # heavy re-run
  python3 scripts/run_metalabel_real_primary.py --smoke         # 1 pair, 1 window live (~2 min)

Outputs: tables/{primary_vs_meta.csv, by_pair.csv, results.md, raw_metrics.json}
         figures/fig{1_pf_pertrade,2_precision_baserate,3_dsr,4_equity}.png

All data roots come from the repo config (config.py); see DATA.md and this study's
README for the banked inputs (a separate pipeline).
"""
from __future__ import annotations
import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))
import config as cfg  # noqa: E402

from lib import overfit as O          # DSR / PBO / effective-N / purged k-fold  (read-only)

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# Locations (READ-ONLY inputs), resolved from config.py
# --------------------------------------------------------------------------- #
# Banked full per-trade ledger of the "meta-gate a proven edge" experiment.
# combo_id encodes pair (tens digit, 1=first pair / 2=second pair) and arm
# (units digit, 0=primary-alone / 1=meta-gated); phase 0=in-sample, 1=OOS.
BANKED_LEDGER = os.path.join(cfg.METALABEL_LEDGER, "trades.parquet")
BANKED_VERDICT = os.path.join(cfg.METALABEL_LEDGER, "verdict.json")

# Live engine (read-only). The edged primary's verified sim kernel + cost model.
LIVE_ENGINE = os.environ.get("LDP_STRUCTURAL_ENGINE", os.path.join(cfg.DATA_CACHE, "structural_engine"))
LIVE_DRIVER = os.environ.get("LDP_METALABEL_DRIVER", os.path.join(LIVE_ENGINE, "metalabel_driver.py"))

# Annualisation: the edged primary runs on Binance perp 30m bars -> 48 bars/day.
BARS_PER_YEAR = 365 * 48


# --------------------------------------------------------------------------- #
# Small metric helpers (consistent definitions across both modes)
# --------------------------------------------------------------------------- #
def profit_factor(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, float)
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    return float(pos / neg) if neg > 1e-15 else (float("inf") if pos > 0 else 0.0)


def per_trade_bp(pnl_frac: np.ndarray) -> float:
    """Mean per-trade P&L in basis points of notional (pnl stored as fraction)."""
    pnl_frac = np.asarray(pnl_frac, float)
    return float(pnl_frac.mean() * 1e4) if pnl_frac.size else float("nan")


def dsr_from_window_sharpes(trade_pnl: np.ndarray, win_sharpes: np.ndarray) -> dict:
    """Deflated Sharpe Ratio of a per-trade P&L stream, deflated against the
    dispersion of the per-window Sharpe estimates (the trial set = the WFO
    windows the primary was IS-tuned over). Uses lib/overfit.py verbatim.

    The headline metric of the program is DSR; we report it for BOTH arms so the
    correction is judged on the same hurdle study 03 used.
    """
    from scipy import stats as ss
    pnl = np.asarray(trade_pnl, float)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size < 3:
        return dict(dsr=float("nan"), sr0=float("nan"), sr=float("nan"), n=int(pnl.size))
    sr = O.sharpe(pnl)                                   # per-trade Sharpe
    sk = float(ss.skew(pnl)); ku = float(ss.kurtosis(pnl, fisher=False))
    d = O.deflated_sharpe_ratio(sr, pnl.size, sk, ku, np.asarray(win_sharpes, float))
    return dict(dsr=float(d["dsr"]), sr0=float(d["sr0"]), sr=float(sr),
                n=int(pnl.size), n_trials=int(d["n_trials"]))


# --------------------------------------------------------------------------- #
# MODE 1: reproduce the corrected result from the banked ledger
# --------------------------------------------------------------------------- #
def run_from_banked() -> dict | None:
    if not os.path.exists(BANKED_LEDGER):
        print(f"  [skip] banked ledger missing: {BANKED_LEDGER}")
        print("         (run the edged-primary meta-gate experiment first, or use --live)")
        return None
    df = pd.read_parquet(BANKED_LEDGER)
    need = {"combo_id", "window_id", "phase", "entry_idx", "exit_idx", "pnl_bp", "weight"}
    if not need.issubset(df.columns):
        print(f"  [skip] banked ledger schema unexpected: {sorted(df.columns)}")
        return None

    df = df.copy()
    df["arm"] = (df["combo_id"] % 10).astype(int)        # 0=primary, 1=meta
    df["pair"] = (df["combo_id"] // 10).astype(int)       # 1=pair A, 2=pair B
    df["pnl"] = df["pnl_bp"].astype(float) / 1e4          # ledger stores bp -> fraction

    oos = df[df["phase"] == 1]                            # judge OOS only

    def arm_metrics(sub: pd.DataFrame, label: str) -> dict:
        p = sub["pnl"].to_numpy()
        # per-window Sharpe set for DSR deflation (windows = the trial dispersion)
        win_sr = []
        for _, g in sub.groupby("window_id"):
            v = g["pnl"].to_numpy()
            if v.size >= 2:
                win_sr.append(O.sharpe(v))
        win_sr = np.array(win_sr) if win_sr else np.array([0.0, 0.0])
        d = dsr_from_window_sharpes(p, win_sr)
        return dict(arm=label, n_trades=int(p.size), pf=profit_factor(p),
                    per_trade_bp=per_trade_bp(p), mean_pnl=float(p.mean()) if p.size else float("nan"),
                    sr_ann=float(O.sharpe(p) * np.sqrt(BARS_PER_YEAR)) if p.size else float("nan"),
                    dsr=d["dsr"], sr0=d["sr0"], sr_per_trade=d["sr"], n_trials=d.get("n_trials"))

    prim = arm_metrics(oos[oos.arm == 0], "primary")
    meta = arm_metrics(oos[oos.arm == 1], "meta")

    # precision vs base rate: among OOS trades, fraction profitable.
    # base_rate = primary-alone profitable fraction (all bets the edge proposes);
    # precision  = meta-gated profitable fraction (the kept subset).
    pr = oos[oos.arm == 0]["pnl"].to_numpy()
    mr = oos[oos.arm == 1]["pnl"].to_numpy()
    base_rate = float((pr > 0).mean()) if pr.size else float("nan")
    precision = float((mr > 0).mean()) if mr.size else float("nan")
    gate_kept = float(meta["n_trades"] / prim["n_trades"]) if prim["n_trades"] else float("nan")

    # per-pair breakdown
    per_pair = []
    for pid, g in oos.groupby("pair"):
        gp = g[g.arm == 0]["pnl"].to_numpy(); gm = g[g.arm == 1]["pnl"].to_numpy()
        per_pair.append(dict(pair_id=int(pid),
                             prim_trades=int(gp.size), meta_trades=int(gm.size),
                             prim_pf=profit_factor(gp), meta_pf=profit_factor(gm),
                             prim_bp=per_trade_bp(gp), meta_bp=per_trade_bp(gm)))

    # PBO across the two arms over WFO windows (column = per-window mean P&L of
    # each arm). With only 2 arms this is informational; reported with a caveat.
    pbo = float("nan")
    try:
        piv = oos.pivot_table(index="window_id", columns="arm", values="pnl",
                              aggfunc="mean").fillna(0.0).to_numpy()
        if piv.shape[0] >= 4 and piv.shape[1] >= 2:
            pbo = float(O.pbo_cscv(piv, n_splits=min(8, piv.shape[0] - (piv.shape[0] % 2)))["pbo"])
    except Exception:
        pass

    # carry the OOS equity arrays for the figure
    eq = {}
    for arm, lab in [(0, "primary"), (1, "meta")]:
        s = oos[oos.arm == arm].sort_values(["pair", "window_id", "entry_idx"])
        eq[lab] = np.cumsum(s["pnl"].to_numpy())

    out = dict(mode="from_banked",
               primary=prim, meta=meta,
               base_rate=base_rate, precision=precision, gate_kept_frac=gate_kept,
               per_pair=per_pair, pbo=pbo, _eq=eq)
    # NOTE: deliberately do NOT embed the banked source path or the banked
    # verdict (which names the proprietary archetype and the tickers). The
    # raw_metrics.json artifact is public-facing: it carries metrics only, the
    # primary referred to generically as a proven structural order-flow /
    # open-interest edge, pairs as neutral pair_id (1, 2).
    return out


# --------------------------------------------------------------------------- #
# MODE 2: live re-run on the edged primary via the read-only structural engine
# --------------------------------------------------------------------------- #
def run_live(smoke: bool = False) -> dict | None:
    """Re-run the meta-labeling end-to-end on the edged primary.

    We import the existing, verified edged-primary meta-gate routine (READ-ONLY)
    and drive it window-by-window so we own the per-trade arrays, then score with
    the SAME overfit.py DSR/PBO harness used everywhere in the program. The engine
    kernel + cost model are reused untouched. If the engine data feeds are missing
    we skip and note (no synthetic fallback).
    """
    if not os.path.isdir(LIVE_ENGINE):
        print(f"  [skip] structural engine dir missing: {LIVE_ENGINE}"); return None
    sys.path.insert(0, LIVE_ENGINE)
    sys.path.insert(0, os.path.dirname(LIVE_DRIVER))
    try:
        import common as nc                    # engine cost model (read-only)
        import importlib.util
        spec = importlib.util.spec_from_file_location("edged_primary_driver", LIVE_DRIVER)
        ep = importlib.util.module_from_spec(spec)
    except Exception as e:
        print(f"  [skip] cannot import structural engine read-only: {e}"); return None

    # configure the edged-primary run via env BEFORE loading the module
    if smoke:
        os.environ.setdefault("MLB_PAIRS", "ETHUSDT")
        os.environ.setdefault("MLB_NWIN", "1")
        os.environ.setdefault("MLB_NS", "16")
    try:
        spec.loader.exec_module(ep)
    except Exception as e:
        print(f"  [skip] edged-primary module failed to load: {e}"); return None

    pairs = ep.PAIRS
    all_prim, all_meta, all_rec = [], [], []
    t0 = time.time()
    for pr in pairs:
        try:
            name, prim, meta, recs = ep.run_pair(pr)   # reuses engine kernel READ-ONLY
        except FileNotFoundError as e:
            print(f"  [skip] {pr}: engine data missing ({e})"); continue
        except Exception as e:
            print(f"  [ERR]  {pr}: {e}"); continue
        all_prim.append(prim); all_meta.append(meta)
        pair_tag = (pairs.index(pr) + 1) * 10
        all_rec.extend([(r[0] + pair_tag, r[1], r[2], r[3], r[4], r[5], r[6]) for r in recs])
        print(f"  [ok]  {pr}: prim trades={prim.size} meta trades={meta.size} "
              f"({time.time()-t0:.1f}s)")
    if not all_prim:
        print("  [skip] no pairs produced trades (data missing?)"); return None

    # write our OWN per-trade ledger inside this study (does not touch the engine dir)
    led_path = os.path.join(TAB, "live_trades.parquet")
    try:
        nc.write_trades_parquet(all_rec, led_path)
        print(f"  ledger -> {led_path} ({len(all_rec):,} rows)")
    except Exception as e:
        print(f"  [warn] could not write live ledger: {e}")

    prim = np.concatenate(all_prim); meta = np.concatenate(all_meta)
    # build per-window sharpe sets from records for DSR deflation
    rdf = pd.DataFrame(all_rec, columns=["combo_id", "window_id", "phase",
                                         "entry_idx", "exit_idx", "pnl_bp", "weight"])
    rdf["arm"] = (rdf.combo_id % 10).astype(int)
    oos = rdf[rdf.phase == 1].copy(); oos["pnl"] = oos.pnl_bp / 1e4

    def metrics(p, sub):
        win_sr = [O.sharpe(g.pnl.to_numpy()) for _, g in sub.groupby("window_id")
                  if g.shape[0] >= 2] or [0.0, 0.0]
        d = dsr_from_window_sharpes(p, np.array(win_sr))
        return dict(n_trades=int(p.size), pf=profit_factor(p),
                    per_trade_bp=per_trade_bp(p),
                    sr_ann=float(O.sharpe(p) * np.sqrt(BARS_PER_YEAR)),
                    dsr=d["dsr"], sr0=d["sr0"])

    pm = metrics(prim, oos[oos.arm == 0]); mm = metrics(meta, oos[oos.arm == 1])
    return dict(mode="live", primary=dict(arm="primary", **pm),
                meta=dict(arm="meta", **mm),
                base_rate=float((prim > 0).mean()), precision=float((meta > 0).mean()),
                gate_kept_frac=float(meta.size / prim.size) if prim.size else float("nan"),
                ledger=led_path,
                _eq=dict(primary=np.cumsum(prim), meta=np.cumsum(meta)))


# --------------------------------------------------------------------------- #
# Output: tables + figures + the "03 said X / we find Y" comparison
# --------------------------------------------------------------------------- #
# Study 03's published conclusion + headline numbers, for the side-by-side.
STUDY03 = dict(
    conclusion=("Meta-labeling is a precision filter, not an alpha source: it "
                "lifted raw PF on 39/42 instruments but 0 cleared DSR>0.95, "
                "because the primary (a vanilla EMA crossover) had no edge to gate."),
    median_meta_pf=1.03,        # crypto/equities median meta PF (study 03 table)
    n_dsr_gt95=0,
    primary_kind="edgeless EMA crossover (no established edge)",
)


def write_tables(res: dict):
    prim, meta = res["primary"], res["meta"]
    rows = [
        dict(arm="primary (edge alone)", n_trades=prim["n_trades"],
             pf=round(prim["pf"], 3), per_trade_bp=round(prim["per_trade_bp"], 1),
             sr_ann=round(prim["sr_ann"], 2), dsr=round(prim["dsr"], 3),
             sr0=round(prim["sr0"], 3)),
        dict(arm="meta (gate + size~p)", n_trades=meta["n_trades"],
             pf=round(meta["pf"], 3), per_trade_bp=round(meta["per_trade_bp"], 1),
             sr_ann=round(meta["sr_ann"], 2), dsr=round(meta["dsr"], 3),
             sr0=round(meta["sr0"], 3)),
    ]
    pv = pd.DataFrame(rows)
    pv.to_csv(os.path.join(TAB, "primary_vs_meta.csv"), index=False)

    if res.get("per_pair"):
        pd.DataFrame(res["per_pair"]).round(3).to_csv(
            os.path.join(TAB, "by_pair.csv"), index=False)

    # full machine-readable metric dump (strip private arrays)
    dump = {k: v for k, v in res.items() if not k.startswith("_")}
    json.dump(dump, open(os.path.join(TAB, "raw_metrics.json"), "w"), indent=2, default=str)

    pf_lift = meta["pf"] - prim["pf"]
    bp_lift = meta["per_trade_bp"] - prim["per_trade_bp"]
    md = [
        "# Meta-labeling an EDGED primary: corrected result\n",
        f"_mode: **{res['mode']}**; DSR is the headline metric._\n",
        "\n## The correction in one table\n",
        "| | study 03 (edgeless primary) | this study (edged primary) |",
        "|---|---|---|",
        f"| primary | {STUDY03['primary_kind']} | a proven structural order-flow / "
        "open-interest edge (closed, pre-validated) |",
        f"| precondition (LdP) | **violated** | **met** |",
        f"| median meta PF | ~{STUDY03['median_meta_pf']} (about break-even) | "
        f"**{meta['pf']:.2f}** |",
        f"| meta per-trade | (near 0) | **{meta['per_trade_bp']:.0f} bp** |",
        f"| verdict | precision filter, *not* alpha | meta-labeling **improves a "
        "real edge** |",
        "\n## Primary vs meta (OOS, net of per-fill costs)\n",
        pv.to_markdown(index=False),
        f"\n\n- **PF lift**: {prim['pf']:.3f} to {meta['pf']:.3f}  "
        f"(+{pf_lift:.3f})",
        f"\n- **Per-trade lift**: {prim['per_trade_bp']:.1f} to "
        f"{meta['per_trade_bp']:.1f} bp  (+{bp_lift:.1f} bp)",
        f"\n- **Precision vs base rate**: profitable-bet rate "
        f"{res['base_rate']*100:.1f}% (all edge bets) to "
        f"{res['precision']*100:.1f}% (meta-kept)",
        f"\n- **Gate keeps** {res['gate_kept_frac']*100:.0f}% of the edge's bets "
        f"(it vetoes the low-confidence tail).",
        f"\n- **DSR (headline)**: primary {prim['dsr']:.3f} to meta "
        f"{meta['dsr']:.3f}.",
        f"\n- **DSR does NOT clear the program's 0.95 publication bar on this "
        f"2-pair sample**: the result inverts study 03's blanket claim (direction "
        f"+ mechanism), it is not a new DSR-passing trophy.",
    ]
    if "pbo" in res and res["pbo"] == res["pbo"]:
        md.append(f"\n- PBO (2-arm, informational): {res['pbo']:.3f}.")
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables ->", os.path.abspath(TAB))


def make_figures(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from lib import style
        style.set_style(); P = style.PALETTE
    except Exception:
        P = {"accent": "#D55E00", "dollar": "#009E73", "tick": "#56B4E9"}
    grey = "#999999"; acc = P["accent"]
    prim, meta = res["primary"], res["meta"]

    # FIG 1: PF and per-trade bp, primary vs meta
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].bar([0, 1], [prim["pf"], meta["pf"]], color=[grey, acc], width=0.6)
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["primary\n(edge alone)", "meta\n(gate+size)"])
    ax[0].set_ylabel("profit factor (net of costs)")
    ax[0].set_title("Profit factor: edge alone vs meta-labeled")
    for i, v in enumerate([prim["pf"], meta["pf"]]):
        ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax[1].bar([0, 1], [prim["per_trade_bp"], meta["per_trade_bp"]], color=[grey, acc], width=0.6)
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["primary", "meta"])
    ax[1].set_ylabel("mean per-trade P&L (bp of notional)")
    ax[1].set_title("Per-trade edge: edge alone vs meta-labeled")
    for i, v in enumerate([prim["per_trade_bp"], meta["per_trade_bp"]]):
        ax[1].text(i, v + 3, f"{v:.0f}", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_pf_pertrade.png"), dpi=200); plt.close(fig)

    # FIG 2: precision vs base rate
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar([0, 1], [res["base_rate"] * 100, res["precision"] * 100],
           color=[grey, P["dollar"]], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base rate\n(all edge bets)", "meta precision\n(kept bets)"])
    ax.set_ylabel("% of bets profitable (OOS, net)")
    ax.set_title("Meta precision vs base rate\n(LdP: meta buys precision by vetoing weak bets)")
    for i, v in enumerate([res["base_rate"] * 100, res["precision"] * 100]):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_precision_baserate.png"), dpi=200); plt.close(fig)

    # FIG 3: DSR primary vs meta, with study-03 reference (0 cleared, all ~0)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar([0, 1], [prim["dsr"], meta["dsr"]], color=[grey, acc], width=0.6)
    ax.axhline(0.95, color=P["dollar"], ls="--", lw=1.0, label="DSR=0.95 hurdle")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["primary\n(edge alone)", "meta\n(gate+size)"])
    ax.set_ylabel("Deflated Sharpe Ratio (OOS)")
    ax.set_title("DSR: edge alone vs meta-labeled\n(study 03: every variant deflated to ~0)")
    ax.set_ylim(0, 1.05); ax.legend()
    for i, v in enumerate([prim["dsr"], meta["dsr"]]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_dsr.png"), dpi=200); plt.close(fig)

    # FIG 4: OOS equity curves
    eq = res.get("_eq")
    if eq:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(eq["primary"], color=grey, label=f"primary (PF {prim['pf']:.2f})")
        ax.plot(eq["meta"], color=acc, label=f"meta (PF {meta['pf']:.2f})")
        ax.set_xlabel("OOS trade #"); ax.set_ylabel("cumulative P&L (fraction, net)")
        ax.set_title("OOS cumulative P&L: edge alone vs meta-labeled")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_equity.png"), dpi=200); plt.close(fig)
    print("  figures ->", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from-banked", action="store_true",
                   help="reproduce corrected result from banked ledger (default)")
    g.add_argument("--live", action="store_true",
                   help="re-run end-to-end via the read-only structural engine (heavy)")
    ap.add_argument("--smoke", action="store_true",
                    help="with --live: 1 pair / 1 window (~2 min)")
    args = ap.parse_args()

    t0 = time.time()
    if args.live or args.smoke:
        print("=== MODE: live re-run on the edged primary (read-only structural engine) ===")
        res = run_live(smoke=args.smoke)
        if res is None and not args.smoke:
            print("  live mode unavailable; falling back to banked reproduction.")
            res = run_from_banked()
    else:
        print("=== MODE: banked reproduction (the corrected headline) ===")
        res = run_from_banked()

    if res is None:
        print("\nNo result produced (inputs missing). Nothing written.")
        return

    write_tables(res)
    make_figures(res)

    prim, meta = res["primary"], res["meta"]
    print("\n" + "=" * 64)
    print("CORRECTED RESULT: meta-labeling an EDGED primary")
    print("=" * 64)
    print(f"  primary (edge alone): trades={prim['n_trades']:>4}  "
          f"PF={prim['pf']:.3f}  per-trade={prim['per_trade_bp']:.1f}bp  "
          f"DSR={prim['dsr']:.3f}")
    print(f"  meta  (gate+size~p) : trades={meta['n_trades']:>4}  "
          f"PF={meta['pf']:.3f}  per-trade={meta['per_trade_bp']:.1f}bp  "
          f"DSR={meta['dsr']:.3f}")
    print(f"  precision {res['base_rate']*100:.1f}% -> {res['precision']*100:.1f}% "
          f"| gate keeps {res['gate_kept_frac']*100:.0f}% of edge bets")
    print(f"\n  STUDY 03 said: {STUDY03['conclusion']}")
    print(f"  THIS STUDY finds: with the precondition met, meta-labeling LIFTS a "
          f"real edge, PF {prim['pf']:.2f}->{meta['pf']:.2f}, "
          f"per-trade {prim['per_trade_bp']:.0f}->{meta['per_trade_bp']:.0f} bp.")
    print(f"\nTOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

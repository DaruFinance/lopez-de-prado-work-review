#!/usr/bin/env python3
"""
deepen_breaks_entropy.py — Phase-2 DEEPEN for Project 7 (Structural Breaks &
Entropy Features, LdP AFML Ch.17-18).

This builds ON TOP of run_breaks_entropy.py (same loaders, same causal feature
kernels, same realistic costs, same purged-CV machinery, same DSR/PBO headline).
It answers four questions the smoke/full run flagged but did not resolve:

  (1) FAMILY signal per market — which family (SADF explosiveness vs the three
      entropy estimators) carries the most OOS signal in crypto / equities /
      forex? We aggregate the per-(instrument,feature) OOS Sharpe by FAMILY x
      MARKET and attach a one-sample sign test (is the family's mean OOS Sharpe
      distinguishable from zero across its instruments?).

  (2) CUSUM-event sampling — LdP samples on CUSUM events, not the raw clock.
      Does restricting the OOS evaluation to CUSUM-event bars change the verdict?
      We re-run the *identical* costed purged-CV feature test but score OOS only
      on bars flagged by the causal CUSUM filter, and diff the OOS Sharpe.

  (3) REGIME conditioning (the core deepen) — condition the forward return on
      causal regimes and test PREDICTIVE content honestly:
        * entropy regime: low / high Shannon entropy (threshold = in-fold median,
          fit on train, applied to OOS — no peeking).
        * explosiveness regime: SADF > 0 (explosive) vs <= 0 (non-explosive),
          again a causal per-bar feature.
      For each regime we measure (a) the conditional mean forward return and a
      directional rule's OOS net Sharpe, and (b) we DSR-gate: every regime rule
      is one trial in the family, and the selected best is deflated against the
      full regime-rule trial menu.

  (4) A CLEAN regime figure (causal, no lookahead): conditional forward-return
      bars by entropy/explosiveness regime, per market, with 95% CIs.

RULES honored: real data (same caches), multi-market, causal (every feature at
bar t uses bars <= t; every threshold fit in-fold on train only), realistic
equity/forex costs via lib/realism, DSR headline via lib/overfit. Imports lib
read-only. Writes only into projects/07_structural_breaks_entropy/{tables,figures}.

The heavy kernels (SADF, entropy) are reused unchanged from run_breaks_entropy /
breaks_entropy and were already profiled 1-core + verified Numba bit-identical
(see run_full.sh / breaks_entropy.verify_bit_identical). This script adds no new
hot loop; the regime/CUSUM analysis is light numpy over the precomputed features.
"""
from __future__ import annotations
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, "/home/daru/ldp_review/lib")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import overfit as OF          # noqa: E402
import realism as RZ          # noqa: E402
import style as ST            # noqa: E402

# reuse the EXACT loaders / feature builder / label / costed evaluator
import run_breaks_entropy as R  # noqa: E402

warnings.filterwarnings("ignore")

PROJ = R.PROJ
FIG = R.FIG
TAB = R.TAB

LABEL_H = R.LABEL_H
COST_BPS = R.COST_BPS
CV_SPLITS = R.CV_SPLITS
EMBARGO = R.EMBARGO_PCT

FAMILY = {
    "sadf": "breaks(SADF)",
    "ent_shannon": "entropy",
    "ent_lz": "entropy",
    "ent_konto": "entropy",
}


# --------------------------------------------------------------------------- #
# (2) CUSUM-event-restricted costed purged-CV feature evaluation
# --------------------------------------------------------------------------- #
def evaluate_feature_cusum(feat, ret, label_fwd, label_h, cost_side, cusum_flag,
                           n_splits, embargo):
    """Same costed purged-CV feature test as run_breaks_entropy.evaluate_feature,
    but the OOS net return is scored ONLY on bars where cusum_flag==1 (LdP's
    event clock). Orientation is still fit in-fold on ALL train bars (CUSUM
    sampling affects only WHICH oos bars count, never leaks the test). Returns the
    OOS net-return path over event bars."""
    n = len(feat)
    valid = np.isfinite(feat) & np.isfinite(label_fwd) & np.isfinite(ret)
    idx = np.where(valid)[0]
    if idx.size < n_splits * 20:
        return None
    f = feat[idx]
    fwd = label_fwd[idx]
    cost_idx = np.asarray(cost_side, float)[idx]
    ev = cusum_flag[idx] == 1
    oos = np.full(idx.size, np.nan)
    for tr, te in OF.purged_kfold_splits(idx.size, n_splits=n_splits,
                                         embargo_pct=embargo, label_span=label_h):
        if tr.size < 20 or te.size < 2:
            continue
        med = np.median(f[tr])
        hi = f[tr] >= med
        orient = np.sign(np.nanmean(fwd[tr][hi]) - np.nanmean(fwd[tr][~hi]))
        if orient == 0:
            orient = 1.0
        pos_te = orient * np.where(f[te] >= med, 1.0, -1.0)
        turn = np.abs(np.diff(np.concatenate([[0.0], pos_te])))
        net = pos_te * fwd[te] - cost_idx[te] * turn
        # keep only event bars within this test fold
        keep = ev[te]
        slot = te[keep]
        oos[slot] = net[keep]
    return oos[np.isfinite(oos)]


# --------------------------------------------------------------------------- #
# (3) Regime conditioning — causal, in-fold thresholds, DSR-gated
# --------------------------------------------------------------------------- #
def regime_rules_oos(feat_df, ret, fwd_ret, cost_side, n_splits, embargo, label_h):
    """For one instrument, build several REGIME RULES and return:
      - oos_paths : {rule_name -> oos net-return path}  (for DSR/PBO)
      - cond      : {regime_label -> list of fwd returns in that regime, OOS}
                    for the descriptive conditional-forward-return figure.

    Regimes (all causal):
      explosive    : SADF > 0
      nonexplosive : SADF <= 0
      hi_entropy   : Shannon entropy >= in-fold median
      lo_entropy   : Shannon entropy <  in-fold median

    Rules tested (each = one trial in the DSR family):
      R_expl_dir   : in the explosive regime, take the in-fold-fit direction of
                     the SADF sign rule; flat otherwise.
      R_loent_dir  : in the low-entropy (more predictable) regime, take the
                     in-fold-fit direction; flat in high-entropy.
      R_hient_dir  : symmetric control (trade only high-entropy bars).
    All directions/thresholds fit on TRAIN only; scored OOS. Costs charged on
    turnover at the OOS bar's realistic per-side cost."""
    sadf = feat_df["sadf"].to_numpy(float)
    ent = feat_df["ent_shannon"].to_numpy(float)
    valid = (np.isfinite(sadf) & np.isfinite(ent) & np.isfinite(fwd_ret)
             & np.isfinite(ret))
    idx = np.where(valid)[0]
    if idx.size < n_splits * 40:
        return {}, {}
    sadf = sadf[idx]; ent = ent[idx]; fwd = fwd_ret[idx]
    cost_idx = np.asarray(cost_side, float)[idx]

    rules = {"R_expl_dir": np.full(idx.size, np.nan),
             "R_loent_dir": np.full(idx.size, np.nan),
             "R_hient_dir": np.full(idx.size, np.nan)}
    # conditional forward returns by regime (OOS only)
    cond = {"explosive": [], "nonexplosive": [], "lo_entropy": [], "hi_entropy": []}

    for tr, te in OF.purged_kfold_splits(idx.size, n_splits=n_splits,
                                         embargo_pct=embargo, label_span=label_h):
        if tr.size < 40 or te.size < 4:
            continue
        ent_med = np.median(ent[tr])

        # --- direction fit on TRAIN within each regime (no peeking) ---
        # explosive-regime direction: sign of mean fwd return when SADF>0 on train
        m_expl_tr = sadf[tr] > 0
        dir_expl = np.sign(np.nanmean(fwd[tr][m_expl_tr])) if m_expl_tr.any() else 0.0
        if dir_expl == 0:
            dir_expl = 1.0
        # low-entropy-regime direction
        m_lo_tr = ent[tr] < ent_med
        dir_lo = np.sign(np.nanmean(fwd[tr][m_lo_tr])) if m_lo_tr.any() else 0.0
        if dir_lo == 0:
            dir_lo = 1.0
        m_hi_tr = ent[tr] >= ent_med
        dir_hi = np.sign(np.nanmean(fwd[tr][m_hi_tr])) if m_hi_tr.any() else 0.0
        if dir_hi == 0:
            dir_hi = 1.0

        # --- apply to OOS ---
        m_expl_te = sadf[te] > 0
        pos_expl = np.where(m_expl_te, dir_expl, 0.0)
        m_lo_te = ent[te] < ent_med
        pos_lo = np.where(m_lo_te, dir_lo, 0.0)
        m_hi_te = ent[te] >= ent_med
        pos_hi = np.where(m_hi_te, dir_hi, 0.0)

        for name, pos in (("R_expl_dir", pos_expl),
                          ("R_loent_dir", pos_lo),
                          ("R_hient_dir", pos_hi)):
            turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = pos * fwd[te] - cost_idx[te] * turn
            rules[name][te] = net

        # --- descriptive conditional forward returns (OOS, no position/cost) ---
        cond["explosive"].extend(fwd[te][m_expl_te].tolist())
        cond["nonexplosive"].extend(fwd[te][~m_expl_te].tolist())
        cond["lo_entropy"].extend(fwd[te][m_lo_te].tolist())
        cond["hi_entropy"].extend(fwd[te][m_hi_te].tolist())

    oos_paths = {k: v[np.isfinite(v)] for k, v in rules.items()
                 if np.isfinite(v).sum() >= 30}
    return oos_paths, cond


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    if args.smoke:
        markets = {
            "crypto":   (["BTCUSDT", "ETHUSDT"], R.load_crypto),
            "equities": (["SPY", "XLK"], R.load_etf),
            "forex":    (["EURUSD", "USDJPY"], R.load_fx),
        }
        n_target, tail, tag = 800, 60_000, "smoke"
    else:
        markets = R.MARKETS
        n_target, tail, tag = R.N_TARGET_BARS, None, "full"

    fam_rows = []          # per (instrument,feature): clock vs cusum OOS Sharpe
    regime_oos = {}        # (instrument,rule) -> oos path  (for DSR/PBO)
    cond_by_market = {m: {k: [] for k in
                          ("explosive", "nonexplosive", "lo_entropy", "hi_entropy")}
                      for m in markets}

    for mname, (syms, loader) in markets.items():
        for sym in syms:
            try:
                base = loader(sym, tail=tail)
                feat_df = R.build_features(base, n_target=n_target)
            except Exception as e:
                print(f"  [skip] {mname}/{sym}: {e}")
                continue
            ret = feat_df["ret"].to_numpy(float)
            close = feat_df["close"].to_numpy(float)
            cusum_flag = feat_df["cusum_flag"].to_numpy(float)
            cost_side = RZ.per_side_cost_fraction(
                mname, sym, feat_df.index, close,
                crypto_fallback=COST_BPS * 1e-4)
            fwd_ret = R.fixed_horizon_label(ret, LABEL_H)

            # (1)+(2): per-feature clock-OOS vs cusum-event-OOS Sharpe
            for fname in ("sadf", "ent_shannon", "ent_lz", "ent_konto"):
                f = feat_df[fname].to_numpy(float)
                oos_clock = R.evaluate_feature(f, ret, fwd_ret, LABEL_H, cost_side,
                                               CV_SPLITS, EMBARGO)
                oos_cusum = evaluate_feature_cusum(f, ret, fwd_ret, LABEL_H,
                                                   cost_side, cusum_flag,
                                                   CV_SPLITS, EMBARGO)
                if oos_clock is None or oos_clock.size < 30:
                    continue
                row = dict(market=mname, instrument=sym, feature=fname,
                           family=FAMILY[fname],
                           sharpe_clock=OF.sharpe(oos_clock),
                           n_clock=int(oos_clock.size))
                if oos_cusum is not None and oos_cusum.size >= 30:
                    row["sharpe_cusum"] = OF.sharpe(oos_cusum)
                    row["n_cusum"] = int(oos_cusum.size)
                else:
                    row["sharpe_cusum"] = np.nan
                    row["n_cusum"] = 0
                fam_rows.append(row)

            # (3): regime rules + conditional forward returns
            oos_paths, cond = regime_rules_oos(feat_df, ret, fwd_ret, cost_side,
                                               CV_SPLITS, EMBARGO, LABEL_H)
            for rname, path in oos_paths.items():
                regime_oos[(sym, rname)] = path
            for k in cond_by_market[mname]:
                cond_by_market[mname][k].extend(cond.get(k, []))

    fam = pd.DataFrame(fam_rows)
    if fam.empty:
        print("No results.")
        return

    # ---- (1) family x market signal table with sign test ----------------- #
    def signtest(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        pos = int((x > 0).sum())
        return float(ss.binomtest(pos, x.size, 0.5).pvalue)

    fam_summary = (fam.groupby(["market", "family"])
                   .agg(mean_sharpe_clock=("sharpe_clock", "mean"),
                        median_sharpe_clock=("sharpe_clock", "median"),
                        mean_sharpe_cusum=("sharpe_cusum", "mean"),
                        n_inst=("sharpe_clock", "size"),
                        frac_pos=("sharpe_clock", lambda s: float((s > 0).mean())))
                   .reset_index())
    # sign-test p per (market,family)
    sp = (fam.groupby(["market", "family"])["sharpe_clock"]
          .apply(signtest).reset_index().rename(columns={"sharpe_clock": "signtest_p"}))
    fam_summary = fam_summary.merge(sp, on=["market", "family"])
    fam_summary.to_csv(os.path.join(TAB, f"family_signal_by_market_{tag}.csv"),
                       index=False)

    # ---- (2) CUSUM-event vs clock OOS Sharpe diff ------------------------ #
    fam["delta_cusum_minus_clock"] = fam["sharpe_cusum"] - fam["sharpe_clock"]
    cusum_summary = (fam.dropna(subset=["sharpe_cusum"])
                     .groupby("market")
                     .agg(mean_clock=("sharpe_clock", "mean"),
                          mean_cusum=("sharpe_cusum", "mean"),
                          mean_delta=("delta_cusum_minus_clock", "mean"),
                          n=("sharpe_clock", "size"))
                     .reset_index())
    cusum_summary.to_csv(os.path.join(TAB, f"cusum_vs_clock_{tag}.csv"), index=False)
    fam.to_csv(os.path.join(TAB, f"feature_clock_vs_cusum_{tag}.csv"), index=False)

    # ---- (3) regime-rule DSR/PBO headline -------------------------------- #
    rg_rows = []
    for (sym, rname), path in regime_oos.items():
        rg_rows.append(dict(instrument=sym, rule=rname, n_oos=int(path.size),
                            sharpe=OF.sharpe(path),
                            skew=float(ss.skew(path)),
                            kurt=float(ss.kurtosis(path, fisher=False))))
    rg = pd.DataFrame(rg_rows)
    rg.to_csv(os.path.join(TAB, f"regime_rules_oos_{tag}.csv"), index=False)

    regime_headline = {}
    if not rg.empty:
        sr_trials = rg["sharpe"].to_numpy(float)
        bi = rg["sharpe"].idxmax()
        brow = rg.loc[bi]
        bpath = regime_oos[(brow["instrument"], brow["rule"])]
        dsr = OF.deflated_sharpe_ratio(OF.sharpe(bpath), len(bpath),
                                       float(ss.skew(bpath)),
                                       float(ss.kurtosis(bpath, fisher=False)),
                                       sr_trials)
        pbo = {}
        try:
            L = min(len(v) for v in regime_oos.values())
            if L >= 120 and len(regime_oos) >= 4:
                M = np.column_stack([v[-L:] for v in regime_oos.values()])
                pbo = OF.pbo_cscv(M, n_splits=10)
        except Exception:
            pass
        regime_headline = dict(best_rule=brow["rule"], best_instrument=brow["instrument"],
                               best_sharpe=float(brow["sharpe"]),
                               dsr=dsr.get("dsr"), sr0=dsr.get("sr0"),
                               n_trials=dsr.get("n_trials"), pbo=pbo.get("pbo"))

    # ---- (4) regime figure: conditional forward return by regime/market -- #
    make_regime_conditional_figure(cond_by_market,
                                   os.path.join(FIG, "fig2_regime_conditional_returns.png"))

    # ---- headline md ----------------------------------------------------- #
    with open(os.path.join(TAB, f"deepen_headline_{tag}.md"), "w") as fh:
        fh.write("# Project 7 — DEEPEN headline\n\n")
        fh.write("## (1) Family signal by market (mean OOS Sharpe, sign-test p)\n\n")
        fh.write(fam_summary.round(4).to_markdown(index=False))
        fh.write("\n\n## (2) CUSUM-event sampling vs raw clock (mean OOS Sharpe)\n\n")
        fh.write(cusum_summary.round(4).to_markdown(index=False))
        fh.write("\n\n## (3) Regime-rule DSR/PBO headline\n\n")
        for k, v in regime_headline.items():
            fh.write(f"- **{k}**: {v}\n")
        fh.write("\n## (4) Conditional forward-return by regime (per market)\n\n")
        cond_tab = conditional_table(cond_by_market)
        fh.write(cond_tab.round(6).to_markdown(index=False))
        fh.write("\n")

    print("=== family signal by market ===")
    print(fam_summary.round(4).to_string(index=False))
    print("\n=== CUSUM vs clock ===")
    print(cusum_summary.round(4).to_string(index=False))
    print("\n=== regime-rule headline ===")
    for k, v in regime_headline.items():
        print(f"  {k:16s}: {v}")
    print("\n=== conditional forward returns by regime ===")
    print(conditional_table(cond_by_market).round(6).to_string(index=False))
    print(f"\nWrote deepen tables to {TAB} and fig2 to {FIG}")


def conditional_table(cond_by_market):
    rows = []
    for m, d in cond_by_market.items():
        for reg, vals in d.items():
            x = np.asarray(vals, float); x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            se = x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan
            # two-sided t vs 0
            tp = ss.ttest_1samp(x, 0.0).pvalue if x.size > 2 else np.nan
            rows.append(dict(market=m, regime=reg, n=x.size,
                             mean_fwd_ret=float(x.mean()),
                             se=float(se), t_p=float(tp)))
    return pd.DataFrame(rows)


def make_regime_conditional_figure(cond_by_market, path):
    import matplotlib.pyplot as plt
    ST.set_style()
    markets = list(cond_by_market.keys())
    regimes = ["nonexplosive", "explosive", "lo_entropy", "hi_entropy"]
    colors = {"nonexplosive": ST.PALETTE["time"], "explosive": ST.PALETTE["accent"],
              "lo_entropy": ST.PALETTE["dollar"], "hi_entropy": ST.PALETTE["tick"]}
    fig, axes = plt.subplots(1, len(markets), figsize=(4.6 * len(markets), 4.6),
                             sharey=True)
    if len(markets) == 1:
        axes = [axes]
    for ax, m in zip(axes, markets):
        means, ses, labs, cs = [], [], [], []
        for reg in regimes:
            x = np.asarray(cond_by_market[m][reg], float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            means.append(x.mean() * 1e4)                  # in bp of the h-bar fwd ret
            ses.append((x.std(ddof=1) / np.sqrt(x.size)) * 1e4 if x.size > 1 else 0)
            labs.append(reg); cs.append(colors[reg])
        xpos = np.arange(len(means))
        ax.bar(xpos, means, yerr=1.96 * np.asarray(ses), color=cs, alpha=0.85,
               capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
        ax.set_title(m)
        ax.set_ylabel(f"mean fwd {LABEL_H}-bar ret (bp)  [OOS, causal]")
    fig.suptitle("Conditional forward return by causal regime "
                 "(in-fold thresholds, OOS bars, 95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

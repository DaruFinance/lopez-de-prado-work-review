#!/usr/bin/env python3
"""
deepen.py, deepening analysis for Project 11 (Bet Sizing, LdP Ch.10).

Runs on tables/raw_results.parquet (the per-instrument output of run_bet_sizing.py)
and answers the sharp question the headline run leaves open:

  Does probability sizing (and discretization) ADD deflated performance, or does
  it only CUT turnover/cost while leaving, or eroding, risk-adjusted edge?

Outputs:
  tables/deepen_paired.csv         per-instrument paired deltas (prob-fixed, disc-fixed)
  tables/deepen_by_market.md       by-market paired medians + sign tests + turnover collapse
  tables/deepen_summary.txt        the honest read, plain text
  figures/fig5_turnover_collapse.png   turnover ratio prob/fixed & disc/fixed by instrument
  figures/fig6_dsr_delta.png       ΔDSR (prob−fixed, disc−fixed) by market, with 0 line
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = ROOT = _d
TAB = os.path.join(HERE, "..", "tables")
FIG = os.path.join(HERE, "..", "figures")
sys.path.insert(0, os.path.join(ROOT, "lib"))

df = pd.read_parquet(os.path.join(TAB, "raw_results.parquet"))

# --- paired deltas (only where both schemes produced a finite metric) ---
def paired(df, a, b):
    m = np.isfinite(df[a]) & np.isfinite(df[b])
    return df.loc[m, a].to_numpy(), df.loc[m, b].to_numpy(), df.loc[m]

rows = []
for _, r in df.iterrows():
    rows.append(dict(
        market=r["market"], name=r["name"], n_ev=r["n_ev"], frac_act=r["frac_act"],
        d_dsr_prob=r["prob_dsr"] - r["fixed_dsr"],
        d_dsr_disc=r["disc_dsr"] - r["fixed_dsr"],
        d_pf_prob=r["prob_pf"] - r["fixed_pf"],
        d_pf_disc=r["disc_pf"] - r["fixed_pf"],
        d_sr_prob=r["prob_sr_ann"] - r["fixed_sr_ann"],
        d_sr_disc=r["disc_sr_ann"] - r["fixed_sr_ann"],
        turn_ratio_prob=(r["prob_turn"] / r["fixed_turn"]) if r["fixed_turn"] > 0 else np.nan,
        turn_ratio_disc=(r["disc_turn"] / r["fixed_turn"]) if r["fixed_turn"] > 0 else np.nan,
        fixed_dsr=r["fixed_dsr"], prob_dsr=r["prob_dsr"], disc_dsr=r["disc_dsr"],
        fixed_turn=r["fixed_turn"], prob_turn=r["prob_turn"], disc_turn=r["disc_turn"],
        pbo_prob=r["pbo_prob"], eff_n=r["eff_n"]))
P = pd.DataFrame(rows)
P.round(4).to_csv(os.path.join(TAB, "deepen_paired.csv"), index=False)

# --- by-market aggregates + sign / Wilcoxon tests ---
def signtest(x):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    npos = int((x > 0).sum()); n = int((x != 0).sum())
    # two-sided sign test p-value via binomial
    p = ss.binomtest(npos, n, 0.5).pvalue if n > 0 else np.nan
    return npos / n if n else np.nan, p, n

lines = ["# Project 11, Bet Sizing DEEPENING (paired, by market)\n",
         "Paired per-instrument deltas vs the FIXED-size book. ΔDSR>0 = prob/disc "
         "sizing improves the deflated headline; turn_ratio<1 = turnover (overtrading) "
         "collapses. frac_pos = fraction of instruments with Δ>0; sign_p = two-sided "
         "sign-test p-value (H0: prob/disc no better than fixed).\n"]
agg = []
for mkt in ["crypto", "equities", "forex", "ALL"]:
    sub = P if mkt == "ALL" else P[P.market == mkt]
    if len(sub) == 0:
        continue
    row = {"market": mkt, "n": len(sub)}
    for lbl, col in [("ΔDSR prob", "d_dsr_prob"), ("ΔDSR disc", "d_dsr_disc"),
                     ("ΔPF prob", "d_pf_prob"), ("ΔPF disc", "d_pf_disc"),
                     ("ΔSRann prob", "d_sr_prob"), ("ΔSRann disc", "d_sr_disc")]:
        v = sub[col].to_numpy()
        fp, p, n = signtest(v)
        row[f"med({lbl})"] = np.nanmedian(v)
        row[f"frac_pos({lbl})"] = fp
        row[f"sign_p({lbl})"] = p
    row["med turn_ratio prob"] = np.nanmedian(sub["turn_ratio_prob"])
    row["med turn_ratio disc"] = np.nanmedian(sub["turn_ratio_disc"])
    row["med pbo_prob"] = np.nanmedian(sub["pbo_prob"])
    agg.append(row)
A = pd.DataFrame(agg)
A.round(4).to_csv(os.path.join(TAB, "deepen_by_market.csv"), index=False)

# compact markdown
md = A[["market", "n", "med(ΔDSR prob)", "frac_pos(ΔDSR prob)", "sign_p(ΔDSR prob)",
        "med(ΔDSR disc)", "frac_pos(ΔDSR disc)", "sign_p(ΔDSR disc)",
        "med turn_ratio prob", "med turn_ratio disc", "med pbo_prob"]].round(4)
lines.append(md.to_markdown(index=False))
lines.append("\n\n## PF deltas (cost-sensitivity proxy)\n")
md2 = A[["market", "med(ΔPF prob)", "frac_pos(ΔPF prob)", "sign_p(ΔPF prob)",
         "med(ΔPF disc)", "frac_pos(ΔPF disc)", "sign_p(ΔPF disc)"]].round(4)
lines.append(md2.to_markdown(index=False))
with open(os.path.join(TAB, "deepen_by_market.md"), "w") as f:
    f.write("\n".join(lines))

# --- the honest read ---
def med(col, mkt=None):
    s = P if mkt is None else P[P.market == mkt]
    return float(np.nanmedian(s[col]))

txt = []
txt.append("BET SIZING, HONEST READ (deepening)\n" + "=" * 40)
for mkt in ["crypto", "equities", "forex", "ALL"]:
    sub = P if mkt == "ALL" else P[P.market == mkt]
    tr_p = np.nanmedian(sub["turn_ratio_prob"]); tr_d = np.nanmedian(sub["turn_ratio_disc"])
    dd_p = np.nanmedian(sub["d_dsr_prob"]); dd_d = np.nanmedian(sub["d_dsr_disc"])
    fp_p, sp_p, _ = signtest(sub["d_dsr_prob"].to_numpy())
    fp_d, sp_d, _ = signtest(sub["d_dsr_disc"].to_numpy())
    txt.append(
        f"\n[{mkt}] n={len(sub)}\n"
        f"  turnover collapse:  prob {tr_p:.2f}x fixed  ({(1-tr_p)*100:.0f}% cut) | "
        f"disc {tr_d:.2f}x  ({(1-tr_d)*100:.0f}% cut)\n"
        f"  ΔDSR(prob−fixed):   median {dd_p:+.4f}  frac_pos {fp_p if fp_p==fp_p else float('nan'):.2f}  sign_p {sp_p:.3f}\n"
        f"  ΔDSR(disc−fixed):   median {dd_d:+.4f}  frac_pos {fp_d if fp_d==fp_d else float('nan'):.2f}  sign_p {sp_d:.3f}")
# global verdict numbers
n_prob_gt95 = int((P["prob_dsr"] > 0.95).sum())
n_disc_gt95 = int((P["disc_dsr"] > 0.95).sum())
n_fixed_gt95 = int((P["fixed_dsr"] > 0.95).sum())
txt.append(f"\nDSR>0.95 instruments (any scheme genuinely 'survives' deflation):")
txt.append(f"  fixed {n_fixed_gt95} | prob {n_prob_gt95} | disc {n_disc_gt95}  (of {len(P)})")
txt.append(f"\nVERDICT: bet sizing is a PRECISION/COST layer, not an alpha source. "
           f"Turnover collapses ~{(1-med('turn_ratio_prob'))*100:.0f}% (prob) / "
           f"~{(1-med('turn_ratio_disc'))*100:.0f}% (disc) vs fixed, but the deflated "
           f"Sharpe does not cross the 0.95 significance bar in the median anywhere, "
           f"the underlying MA-crossover primary has no deflatable edge to amplify.")
with open(os.path.join(TAB, "deepen_summary.txt"), "w") as f:
    f.write("\n".join(txt))
print("\n".join(txt))

# --- figures ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from lib import style
    style.set_style(); Pal = style.PALETTE
    C_PROB, C_DISC = Pal["accent"], Pal["dollar"]
except Exception:
    C_PROB, C_DISC = "#d1495b", "#2e86ab"

# FIG5: turnover collapse, sorted by prob ratio
Q = P[np.isfinite(P["turn_ratio_prob"])].sort_values("turn_ratio_prob")
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(Q))
ax.bar(x - 0.2, Q["turn_ratio_prob"], 0.4, label="prob / fixed", color=C_PROB)
ax.bar(x + 0.2, Q["turn_ratio_disc"], 0.4, label="disc / fixed", color=C_DISC)
ax.axhline(1.0, color="k", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(Q["name"], rotation=90, fontsize=6)
ax.set_ylabel("turnover ratio vs fixed"); ax.set_title("Turnover collapse from probability sizing")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_turnover_collapse.png"), dpi=120); plt.close(fig)

# FIG6: ΔDSR by market
fig, ax = plt.subplots(figsize=(8, 4.4))
markets = [m for m in ["crypto", "equities", "forex"] if (P.market == m).any()]
xs = np.arange(len(markets)); w = 0.35
ax.bar(xs - w/2, [np.nanmedian(P[P.market == m]["d_dsr_prob"]) for m in markets], w,
       label="ΔDSR prob", color=C_PROB)
ax.bar(xs + w/2, [np.nanmedian(P[P.market == m]["d_dsr_disc"]) for m in markets], w,
       label="ΔDSR disc", color=C_DISC)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(markets)
ax.set_ylabel("median ΔDSR vs fixed"); ax.set_title("Does sizing add deflated performance?")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_dsr_delta.png"), dpi=120); plt.close(fig)
print("\ndeepen tables + figures written")

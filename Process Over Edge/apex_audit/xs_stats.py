"""xs_stats.py -- uniform overfitting / rigor layer for the T_XS cross-sectional corpus.

Ports the de Prado overfitting stack (corpus_overfit.py, written for T_ML's
window-aggregated cells) to the T_XS output format, so EVERY ML family (lgbm, mlp,
and the deep zoo to come) is judged by the IDENTICAL, honest gates that were applied
to single-series T_ML. The whole point of the paper's "ML edge vs static edge"
comparison is that the methodology is uniform; this file is that uniformity.

This is READ-ONLY post-analysis of banked ledgers. NO engine edits, NO re-runs.

------------------------------------------------------------------------------------
WHAT THE T_XS FORMAT GIVES US (and how it differs from T_ML)
------------------------------------------------------------------------------------
A T_XS run dir contains some subset of:
  trades.parquet : per-BAR net pnl ledger -> cols combo_id, window_id, phase, bar_idx, pnl_bp
                   phase 0=IS, 1=OOS. bar_idx is the GLOBAL 1h-grid bar index, so streams
                   from different combos are ALIGNABLE on a shared timeline.
  cells.parquet  : per-window aggregates -> combo_id, window_id, phase, n_reb, pnl_bp, pos_bp, neg_bp
  summary.csv    : per-combo OOS roll-up -> combo_id, family, H, windows, oos_bars, oos_sharpe,
                   oos_pf, oos_total_bp, oos_mean_bp
  combos.parquet : combo_id -> family, H

The Modal mlp runs (TXS_modal/mlp_H*) ship ONLY trades.parquet, with combo_id 0 inside
each per-horizon directory; we infer family="mlp" and H from the dir name and reconstruct
cells/combos/summary from the per-bar ledger.

KEY structural fact for honest N_trials: in T_XS, combo_id = ONE id per (family, H). The
bet-sizing knobs (q / long_only / tilt / rebalance throttle) are IS-SELECTED per window,
NOT enumerated as separate combos (the structural-vs-IS-tunable rule). So the number of
combos is NOT the number of trials searched. We expose N_trials explicitly:
    N_trials = (#combos) * KNOBS_SEARCHED      (KNOBS_SEARCHED default 48, = XS_NS_KNOB)
and report DSR under both the combo count and the knob-inflated count, because deflating
by 10 trials when the search actually examined ~480 knob configs would understate the
multiple-testing burden.

------------------------------------------------------------------------------------
THE BIG IMPROVEMENT OVER T_ML's redundancy/Sharpe: PER-BAR STREAMS
------------------------------------------------------------------------------------
T_ML's redundancy/DSR were forced onto ~28-point per-window streams (cells.parquet is
window-aggregated), which sit ON the independent-noise floor (01_redundancy.md): with
~28 points the SE of a Pearson rho is ~0.20, so "uncorrelated" largely measured sampling
noise, and effective-N collapsed to ~40 of 24,887. T_XS writes a per-BAR ledger, so for
T_XS we compute:
  - per-combo OOS Sharpe + DSR on the per-BAR net-bp stream (thousands of points, real SE),
  - redundancy / effective-N by correlating per-BAR OOS net-bp streams on the shared
    bar_idx grid (a stable rho, not a 28-point estimate).
We still report a per-WINDOW DSR variant for apples-to-apples with the T_ML report, and we
flag when a stream is short.

------------------------------------------------------------------------------------
WINDOW COUNT CAVEAT (CSCV needs enough windows)
------------------------------------------------------------------------------------
T_XS_main has only 7 WFO windows. CSCV/PBO (Bailey-Borwein-LopezdePrado-Zhu 2017) needs
S even window-blocks split into C(S,S/2) train/test partitions; with 7 windows the most we
can do is S=6 -> C(6,3)=20 partitions, which is statistically thin and we say so. As the
PRIMARY cross-validation read for so few windows we instead report an IS->OOS rank-
persistence statistic (Spearman rho between per-window IS and OOS performance, pooled over
windows) and report CSCV PBO only as a secondary, explicitly-marginal number. With more
combos/windows (the model zoo + more horizons) PBO becomes the headline again.

------------------------------------------------------------------------------------
METHODS (cited)
------------------------------------------------------------------------------------
- DSR / PSR : Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt.
- PBO / CSCV: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
              Overfitting", J. Computational Finance 20(4).
- Effective number of bets (eigenvalue/entropy of the correlation spectrum): Meucci (2009),
  "Managing Diversification"; we report both ENB = (Sum lam)^2 / Sum lam^2 and the
  exp-entropy of the normalized eigenvalues.

------------------------------------------------------------------------------------
INGESTING THE STATIC BASELINE (T_NOVEL / T6) FOR THE ML-vs-STATIC COMPARISON  [adapter note]
------------------------------------------------------------------------------------
The static corpora live under the structural-edge engine root (config.STRUCTURAL_ENGINE,
NOVEL*/T6* subtrees) with a DIFFERENT layout: one trades.parquet PER (archetype, pair) in
nested dirs (e.g. <NOVEL_run>/<archetype>/<PAIR>USDT/trades.parquet), and those ledgers are
per-TRADE (entry/exit + net bp), not per-bar, and are produced by the structural kernel with
its own column names. To feed them through THIS same gating, write a thin adapter (NOT built
here) that, per static "combo" = (archetype, pair):
  1. maps its per-trade ledger to a per-window pnl series keyed on the SAME WFO window ids
     used in T_XS (or to a per-bar net-bp series on the shared 1h grid by spreading each
     trade's net bp at its exit bar) -> feed `combo_panels` exactly like an ML family,
  2. sets family=archetype, H=NA (static signals have no ML horizon), and
  3. sets N_trials = the static search's structural combo count (e.g. NOVEL's
     sig x confluence x exit grid) so the DSR haircut is apples-to-apples.
Then call `report_runs([...])` with the ML run dirs AND the adapted static dirs to get the
single ML-vs-static table. Only the loader differs; all metrics below are format-agnostic
once a run is reduced to {per-combo per-bar OOS stream, per-window IS/OOS pnl, pos/neg bp,
n_reb, family, H, N_trials}.

------------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------------
  python3 xs_stats.py                              # default: XS_main + the 4 modal mlp dirs
  python3 xs_stats.py /path/run1 /path/run2 ...    # explicit run dirs (each = one family or a
                                                   #   multi-combo dir like XS_main)
Each positional arg is a run DIR. A dir with combos.parquet is multi-combo (XS_main); a dir
named mlp_H* with only trades.parquet is a single (family,H) modal run.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = _d
sys.path.insert(0, REPO_ROOT)
from config import LIB as _LIB
sys.path.insert(0, _LIB)
sys.path.insert(0, HERE)

import math
import glob
import itertools
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
# Banked cross-sectional run dirs read by the default report. These are produced by
# xs_run.py (XS_main) and the managed-accelerator mlp_H* runs; not part of the bundled
# config map, so they resolve from LDP_XS_RUNS, defaulting under the repo data/ dir.
_XS_RUNS = os.environ.get("LDP_XS_RUNS", os.path.join(REPO_ROOT, "data", "xs_runs"))
DEFAULT_MAIN = os.path.join(_XS_RUNS, "XS_main")
DEFAULT_MODAL_GLOB = os.path.join(_XS_RUNS, "mlp_H*")

# Knobs IS-searched per window (q x long_only x tilt x reb). The runner default is
# XS_NS_KNOB=48; the real search space per (family,H) is this many configs, so the
# multiple-testing burden is ~ combos * KNOBS_SEARCHED, not combos.
KNOBS_SEARCHED = int(os.environ.get("XS_NS_KNOB", 48))

BARS_PER_YEAR = 24 * 365          # 8760, 1h bars
CSCV_S_MAX = 10                   # cap; clamped to (even) <= W
DSR_SURV = 0.95                   # headline DSR survivor threshold (paper bar)
REDUND_MAX_COMBOS = 6000          # eigen subsample cap (RAM); we are far below this in T_XS


# ---------------------------------------------------------------------------
# normal CDF / inverse-CDF (Acklam) -- copied from corpus_overfit.py for parity
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---------------------------------------------------------------------------
# Loading -- reduce any T_XS run dir to a uniform per-combo record
# ---------------------------------------------------------------------------
def _infer_family_H_from_dirname(d):
    """For modal mlp_H* dirs: family='mlp', H parsed from the suffix."""
    base = os.path.basename(d.rstrip("/"))
    fam = base.split("_")[0] if "_" in base else base
    H = None
    if "_H" in base:
        try:
            H = int(base.split("_H")[-1])
        except ValueError:
            H = None
    return fam, H


def load_run(run_dir):
    """Load one run dir into a list of per-combo records.

    Returns a list of dicts, one per (combo_id within this dir), each with:
      run        : run_dir basename (for labeling)
      combo_id   : original combo id within the dir
      key        : globally-unique label (run::family::H or run::combo)
      family, H  : taxonomy
      oos_bar_idx, oos_bar_pnl : aligned per-BAR OOS net-bp stream (np arrays, sorted by bar_idx)
      is_win_pnl, oos_win_pnl  : per-WINDOW net-bp (dict window_id->bp) IS / OOS
      oos_pos, oos_neg         : total gross winning / losing bp OOS (for PF)
      n_reb                    : total OOS rebalances (turnover proxy)
      oos_n_bars               : number of OOS bars (for Sharpe sample size)
    """
    tpath = os.path.join(run_dir, "trades.parquet")
    if not os.path.exists(tpath):
        return []
    tr = pd.read_parquet(tpath)
    # combos / family-H lookup
    cpath = os.path.join(run_dir, "combos.parquet")
    if os.path.exists(cpath):
        cdf = pd.read_parquet(cpath)
        fam_of = dict(zip(cdf.combo_id, cdf.family))
        H_of = dict(zip(cdf.combo_id, cdf.H))
    else:
        fam, H = _infer_family_H_from_dirname(run_dir)
        fam_of, H_of = {}, {}
        for c in tr.combo_id.unique():
            fam_of[c] = fam
            H_of[c] = H
    # cells (for pos/neg/n_reb if present; else reconstruct from per-bar ledger)
    cells = None
    clpath = os.path.join(run_dir, "cells.parquet")
    if os.path.exists(clpath):
        cells = pd.read_parquet(clpath)

    runname = os.path.basename(run_dir.rstrip("/"))
    recs = []
    for cid, g in tr.groupby("combo_id"):
        fam = fam_of.get(cid, "?")
        H = H_of.get(cid, None)
        oos = g[g.phase == 1]
        is_ = g[g.phase == 0]
        # aligned per-bar OOS stream (sorted by global bar idx; OOS windows are disjoint in time)
        oos_sorted = oos.sort_values("bar_idx")
        oos_bar_idx = oos_sorted.bar_idx.to_numpy()
        oos_bar_pnl = oos_sorted.pnl_bp.to_numpy().astype(np.float64)
        # per-window net bp
        is_win = is_.groupby("window_id").pnl_bp.sum().astype(np.float64).to_dict()
        oos_win = oos.groupby("window_id").pnl_bp.sum().astype(np.float64).to_dict()
        # pos/neg + n_reb : prefer cells.parquet, else derive from per-bar ledger
        if cells is not None:
            cc = cells[(cells.combo_id == cid) & (cells.phase == 1)]
            oos_pos = float(cc.pos_bp.sum())
            oos_neg = float(cc.neg_bp.sum())
            n_reb = int(cc.n_reb.sum()) if "n_reb" in cc else 0
        else:
            oos_pos = float(oos_bar_pnl[oos_bar_pnl > 0].sum())
            oos_neg = float(-oos_bar_pnl[oos_bar_pnl < 0].sum())
            n_reb = 0
        recs.append(dict(
            run=runname, combo_id=int(cid),
            key=f"{runname}::{fam}::H{H}" if H is not None else f"{runname}::{fam}::c{cid}",
            family=str(fam), H=(int(H) if H is not None else None),
            oos_bar_idx=oos_bar_idx, oos_bar_pnl=oos_bar_pnl,
            is_win_pnl=is_win, oos_win_pnl=oos_win,
            oos_pos=oos_pos, oos_neg=oos_neg, n_reb=n_reb,
            oos_n_bars=int(oos_bar_pnl.size)))
    return recs


# ---------------------------------------------------------------------------
# 1. per-combo headline metrics (PF, Sharpe annualized, %positive windows, turnover)
# ---------------------------------------------------------------------------
def _sharpe_moments(x):
    """Per-bar Sharpe + skew + (raw) kurtosis + n for a 1-D net-bp stream."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return dict(sr=np.nan, n=n, skew=np.nan, kurt=3.0)
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0:
        return dict(sr=np.nan, n=n, skew=np.nan, kurt=3.0)
    z = (x - mu) / sd
    return dict(sr=mu / sd, n=n, skew=float(np.mean(z**3)), kurt=float(np.mean(z**4)))


def combo_headline(rec):
    """PF, per-bar Sharpe, annualized Sharpe, %positive windows, turnover proxy.

    Annualization caveat (TXS_PAPER_PLAN / xs_common): naive per-bar Sharpe *
    sqrt(8760) is OPTIMISTIC because rotation returns are autocorrelated (an H-bar
    hold makes consecutive bars dependent), so the effective independent sample is
    far below the raw bar count. We print the annualized number as indicative and
    treat PF + the per-WINDOW DSR (honest n) as the primary reads."""
    m = _sharpe_moments(rec["oos_bar_pnl"])
    pf = (rec["oos_pos"] / rec["oos_neg"]) if rec["oos_neg"] > 0 else np.nan
    owin = rec["oos_win_pnl"]
    pct_pos = (np.mean([v > 0 for v in owin.values()]) if owin else np.nan)
    sr_bar = m["sr"]
    sr_ann = sr_bar * math.sqrt(BARS_PER_YEAR) if np.isfinite(sr_bar) else np.nan
    # turnover proxy: rebalances per OOS bar (if n_reb available), else NaN
    turn = (rec["n_reb"] / rec["oos_n_bars"]) if rec["n_reb"] and rec["oos_n_bars"] else np.nan
    return dict(pf=pf, sr_bar=sr_bar, sr_ann=sr_ann, pct_pos_win=pct_pos,
                n_bars=m["n"], skew=m["skew"], kurt=m["kurt"], turnover=turn,
                n_win=len(owin))


def _per_window_stream(rec):
    """The per-WINDOW OOS net-bp stream (one obs per WFO window) for honest DSR.

    Per-bar streams overstate sample size under rotation autocorrelation; the
    per-window stream is the conservative, T_ML-comparable observation unit."""
    return np.array(list(rec["oos_win_pnl"].values()), dtype=np.float64)


def family_dsr(recs, N_trials, stream="window"):
    """DSR computed WITHIN a family (its own cross-combo Sharpe dispersion + N).

    This is the correct unit for the paper's per-family table: the deflation
    benchmark SR0 is the expected best-of-N Sharpe given the dispersion of THIS
    family's trials, not a cross-family pool (lgbm and mlp have different scales /
    bar counts, so pooling their Sharpe variance corrupts the benchmark).

    stream='window' -> per-WINDOW Sharpe (honest n, default, T_ML-comparable).
    stream='bar'    -> per-BAR Sharpe (optimistic n; sensitivity only).

    With very few combos per family the cross-combo Var(SR) is itself a noisy
    estimate; we return it so the caller can flag it. Returns
    (dsr_array, sr_array, sr0, var_sr)."""
    streams = [(_per_window_stream(r) if stream == "window" else r["oos_bar_pnl"])
               for r in recs]
    moms = [_sharpe_moments(s) for s in streams]
    sr = np.array([m["sr"] for m in moms])
    n = np.array([m["n"] for m in moms])
    sk = np.array([m["skew"] for m in moms])
    ku = np.array([m["kurt"] for m in moms])
    sr0, var_sr = deflated_sr_benchmark(sr, N_trials)
    if not np.isfinite(sr0):
        return np.full(len(recs), np.nan), sr, sr0, var_sr
    return psr(sr, n, sk, ku, sr0), sr, sr0, var_sr


# ---------------------------------------------------------------------------
# 2. DSR (Deflated Sharpe Ratio)  -- Bailey & Lopez de Prado (2014)
# ---------------------------------------------------------------------------
def psr(sr, n, skew, kurt, sr_star):
    """PSR = P(true SR > sr_star). Moment-aware std error. Scalar or array."""
    sr = np.asarray(sr, float)
    denom = np.sqrt(np.clip(1 - skew * sr + (kurt - 1) / 4.0 * sr * sr, 1e-12, None))
    z = (sr - sr_star) * np.sqrt(np.clip(np.asarray(n, float) - 1, 0, None)) / denom
    if np.ndim(z) == 0:
        return _norm_cdf(z) if np.isfinite(z) else np.nan
    return np.array([_norm_cdf(v) if np.isfinite(v) else np.nan for v in z])


def deflated_sr_benchmark(sr_array, N_trials):
    """SR0 = expected max Sharpe of N independent trials (de Prado 2014).

    SR0 = sqrt(Var(SR_trials)) * [(1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e))], g=Euler-Mascheroni.
    Var taken over the cross-section of trial Sharpes. With few combos this variance is
    itself noisy; we therefore ALSO accept an externally-supplied var (var_override) so the
    deflation can use a stable cross-trial Sharpe dispersion when the family has <~5 combos."""
    s = np.asarray(sr_array, float)
    s = s[np.isfinite(s)]
    if len(s) < 2 or N_trials < 2:
        return float("nan"), (np.var(s, ddof=1) if len(s) >= 2 else float("nan"))
    var_sr = np.var(s, ddof=1)
    g = 0.5772156649
    e = math.e
    sr0 = math.sqrt(var_sr) * ((1 - g) * _norm_ppf(1 - 1.0 / N_trials)
                               + g * _norm_ppf(1 - 1.0 / (N_trials * e)))
    return sr0, var_sr


def dsr_for_records(recs, N_trials, var_sr=None):
    """Per-record DSR using the PER-BAR OOS Sharpe stream (legacy / general helper).

    Kept for the static-baseline adapter and ad-hoc use; the per-family driver uses
    `family_dsr` (which switches per-window vs per-bar and deflates within a family).
    Returns (dsr_array aligned with recs, sr_array, sr0, var_used). If var_sr is supplied
    we deflate against that fixed benchmark variance, else estimate it from this set."""
    moms = [_sharpe_moments(r["oos_bar_pnl"]) for r in recs]
    sr = np.array([m["sr"] for m in moms])
    n = np.array([m["n"] for m in moms])
    sk = np.array([m["skew"] for m in moms])
    ku = np.array([m["kurt"] for m in moms])
    if var_sr is not None and np.isfinite(var_sr) and var_sr > 0:
        g = 0.5772156649
        e = math.e
        sr0 = math.sqrt(var_sr) * ((1 - g) * _norm_ppf(1 - 1.0 / N_trials)
                                   + g * _norm_ppf(1 - 1.0 / (N_trials * e)))
        var_used = var_sr
    else:
        sr0, var_used = deflated_sr_benchmark(sr, N_trials)
    dsr = psr(sr, n, sk, ku, sr0) if np.isfinite(sr0) else np.full(len(recs), np.nan)
    return np.asarray(dsr), sr, sr0, var_used


# ---------------------------------------------------------------------------
# 3a. PBO via CSCV (secondary; marginal at low window count)
# ---------------------------------------------------------------------------
def cscv_pbo(is_mat, oos_mat, S_max=CSCV_S_MAX):
    """PBO on (combos x windows) IS/OOS per-window net-bp matrices (de Prado et al. 2017).

    Same algorithm as corpus_overfit.cscv_pbo. NaN->0. Returns (pbo, n_splits, S_used,
    win_per_block, median_logit_lambda). Needs >=2 combos and S>=2."""
    C, W = is_mat.shape
    if C < 2 or W < 2:
        return float("nan"), 0, 0, 0, float("nan")
    pis = np.nan_to_num(is_mat); pos = np.nan_to_num(oos_mat)
    S = min(S_max, W)
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return float("nan"), 0, S, 0, float("nan")
    wper = W // S
    block_of = np.repeat(np.arange(S), wper)
    cols = np.arange(wper * S)
    is_block = np.zeros((C, S)); oos_block = np.zeros((C, S))
    for b in range(S):
        sel = cols[block_of == b]
        is_block[:, b] = pis[:, sel].sum(1)
        oos_block[:, b] = pos[:, sel].sum(1)
    half = S // 2
    lambdas = []
    for train in itertools.combinations(range(S), half):
        test = [b for b in range(S) if b not in train]
        ip = is_block[:, list(train)].sum(1)
        op = oos_block[:, test].sum(1)
        nstar = int(np.argmax(ip))
        less = np.sum(op < op[nstar]); equal = np.sum(op == op[nstar])
        rank = less + (equal + 1) / 2.0
        w = min(max(rank / (C + 1.0), 1e-6), 1 - 1e-6)
        lambdas.append(math.log(w / (1 - w)))
    lambdas = np.array(lambdas)
    return float(np.mean(lambdas <= 0)), len(lambdas), S, wper, float(np.median(lambdas))


# ---------------------------------------------------------------------------
# 3b. IS->OOS rank-persistence (PRIMARY when windows are too few for CSCV)
# ---------------------------------------------------------------------------
def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_persistence(recs):
    """Pooled IS->OOS rank persistence across combos and windows.

    For each WFO window, rank combos by IS net-bp and by OOS net-bp; Spearman rho per
    window; report the mean and the pooled rho (stacking all (IS_rank, OOS_rank) pairs).
    rho>0 => IS winners tend to stay OOS winners (the opposite of overfitting). This is
    the robust read when there are too few windows for a stable CSCV PBO. Needs >=3 combos."""
    if len(recs) < 3:
        return dict(mean_rho=np.nan, pooled_rho=np.nan, n_win=0, frac_pos=np.nan)
    wins = sorted(set().union(*[set(r["oos_win_pnl"].keys()) for r in recs]))
    per_win = []
    pooled_is, pooled_oos = [], []
    for w in wins:
        isv = np.array([r["is_win_pnl"].get(w, np.nan) for r in recs])
        oov = np.array([r["oos_win_pnl"].get(w, np.nan) for r in recs])
        m = np.isfinite(isv) & np.isfinite(oov)
        if m.sum() < 3:
            continue
        rho = _spearman(isv[m], oov[m])
        if np.isfinite(rho):
            per_win.append(rho)
        pooled_is.append(pd.Series(isv[m]).rank().to_numpy())
        pooled_oos.append(pd.Series(oov[m]).rank().to_numpy())
    if not per_win:
        return dict(mean_rho=np.nan, pooled_rho=np.nan, n_win=0, frac_pos=np.nan)
    pooled = _spearman(np.concatenate(pooled_is), np.concatenate(pooled_oos))
    return dict(mean_rho=float(np.mean(per_win)), pooled_rho=pooled,
                n_win=len(per_win), frac_pos=float(np.mean(np.array(per_win) > 0)))


# ---------------------------------------------------------------------------
# 4. Redundancy / effective-N on PER-BAR OOS streams (shared bar_idx grid)
# ---------------------------------------------------------------------------
def effective_n(recs, min_overlap=200):
    """Median pairwise |rho| + effective number of independent strategies.

    Builds a (combos x bars) matrix on the UNION of OOS bar_idx (pairwise-complete corr,
    NOT zero-filled -- zero-fill biased the T_ML number down ~40%). Per-bar streams give
    thousands of overlapping points so each rho is well-estimated (unlike T_ML's ~28-pt
    per-window streams that sat on the noise floor).

    Effective-N via two estimators on the eigenvalues of the correlation matrix:
      ENB_meucci = (sum lam)^2 / sum lam^2          (Meucci 2009)
      ENB_entropy = exp( -sum p_i ln p_i ), p_i = lam_i / sum lam   (participation entropy)
    Returns dict with median_abs_rho, pct_lt_0.5, ENB_meucci, ENB_entropy, n_combos, caveat.

    SHORT-STREAM CAVEAT (carried from 01_redundancy.md): with very few combos the
    correlation spectrum is trivially low-dimensional; ENB is only meaningful once the
    corpus is wide. We report it honestly with n_combos so the reader can judge."""
    C = len(recs)
    if C < 2:
        return dict(median_abs_rho=np.nan, pct_lt_05=np.nan, enb_meucci=np.nan,
                    enb_entropy=np.nan, n_combos=C, n_pairs=0,
                    caveat="need >=2 combos")
    # union bar grid
    allbars = np.unique(np.concatenate([r["oos_bar_idx"] for r in recs]))
    bpos = {b: i for i, b in enumerate(allbars)}
    B = len(allbars)
    M = np.full((C, B), np.nan)
    for i, r in enumerate(recs):
        idx = np.array([bpos[b] for b in r["oos_bar_idx"]])
        M[i, idx] = r["oos_bar_pnl"]
    # pairwise-complete correlations
    rhos = []
    corr = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            m = np.isfinite(M[i]) & np.isfinite(M[j])
            if m.sum() < min_overlap:
                continue
            xi, xj = M[i, m], M[j, m]
            if np.std(xi) == 0 or np.std(xj) == 0:
                continue
            r_ij = float(np.corrcoef(xi, xj)[0, 1])
            rhos.append(abs(r_ij))
            corr[i, j] = corr[j, i] = r_ij
    if not rhos:
        return dict(median_abs_rho=np.nan, pct_lt_05=np.nan, enb_meucci=np.nan,
                    enb_entropy=np.nan, n_combos=C, n_pairs=0,
                    caveat="no pair met min_overlap")
    med = float(np.median(rhos))
    pct = float(np.mean(np.array(rhos) < 0.5))
    # eigen-based effective N (on the symmetric corr matrix; clip tiny negatives)
    w = np.linalg.eigvalsh(corr)
    w = np.clip(w, 1e-12, None)
    enb_meucci = float((w.sum() ** 2) / (w ** 2).sum())
    p = w / w.sum()
    enb_entropy = float(np.exp(-(p * np.log(p)).sum()))
    return dict(median_abs_rho=med, pct_lt_05=pct, enb_meucci=enb_meucci,
                enb_entropy=enb_entropy, n_combos=C, n_pairs=len(rhos),
                caveat=("thin corpus: ENB only meaningful at scale"
                        if C < 30 else "ok"))


# ---------------------------------------------------------------------------
# Driver / report
# ---------------------------------------------------------------------------
def _is_oos_window_matrices(recs):
    """Stack per-window IS/OOS net-bp into (combos x windows) matrices on a union grid."""
    wins = sorted(set().union(*[set(r["oos_win_pnl"].keys()) | set(r["is_win_pnl"].keys())
                                for r in recs]))
    wi = {w: i for i, w in enumerate(wins)}
    C, W = len(recs), len(wins)
    ism = np.full((C, W), np.nan); oom = np.full((C, W), np.nan)
    for i, r in enumerate(recs):
        for w, v in r["is_win_pnl"].items():
            ism[i, wi[w]] = v
        for w, v in r["oos_win_pnl"].items():
            oom[i, wi[w]] = v
    return ism, oom, wins


def report_runs(run_dirs):
    # ---- load everything into a flat list of per-combo records ----
    all_recs = []
    for d in run_dirs:
        rs = load_run(d)
        if not rs:
            print(f"-- {d}: no trades.parquet, skipped --")
            continue
        all_recs.extend(rs)
    if not all_recs:
        print("no data loaded.")
        return

    # ---- per-combo headline ----
    rows = []
    for r in all_recs:
        h = combo_headline(r)
        rows.append(dict(key=r["key"], family=r["family"], H=r["H"],
                         **h, n_combo_bars=r["oos_n_bars"]))
    hdf = pd.DataFrame(rows)
    n_combos_total = len(all_recs)

    print("=" * 100)
    print(f"T_XS RIGOR REPORT  --  {n_combos_total} combos across {len(run_dirs)} run dir(s)")
    print(f"N_trials convention (DSR haircut): PER FAMILY, N = (#combos in family) x "
          f"KNOBS_SEARCHED ({KNOBS_SEARCHED}).")
    print("  combo_id = one per (family,H); the q/long_only/tilt/reb knobs are IS-SELECTED")
    print(f"  per window, so the real search per (family,H) examined ~{KNOBS_SEARCHED} configs.")
    print("  DSR uses each family's OWN cross-combo Sharpe dispersion (NOT pooled across")
    print("  families -- lgbm and mlp have different scales/bar counts).")
    print("=" * 100)

    # ---- per-family DSR (PRIMARY = per-WINDOW Sharpe, honest n; SENS = per-BAR) ----
    fam_recs = {}
    for r in all_recs:
        fam_recs.setdefault(r["family"], []).append(r)
    fam_dsr_win, fam_dsr_bar, fam_meta = {}, {}, {}
    for fam, frecs in fam_recs.items():
        Nf = len(frecs) * KNOBS_SEARCHED
        dwin, srw, sr0w, varw = family_dsr(frecs, Nf, stream="window")
        dbar, srb, sr0b, varb = family_dsr(frecs, Nf, stream="bar")
        for r, dw, db in zip(frecs, dwin, dbar):
            r["_dsr_win"] = dw
            r["_dsr_bar"] = db
        fam_dsr_win[fam] = np.asarray(dwin)
        fam_dsr_bar[fam] = np.asarray(dbar)
        fam_meta[fam] = dict(N=Nf, n=len(frecs), sr0_win=sr0w, sr0_bar=sr0b,
                             var_win=varw, var_bar=varb)

    # ---- cross-validation of selection ----
    ism, oom, wins = _is_oos_window_matrices(all_recs)
    pbo, nsplit, S_used, wper, medlam = cscv_pbo(ism, oom)
    rp = rank_persistence(all_recs)

    # ---- redundancy / effective-N on per-bar streams ----
    red = effective_n(all_recs)

    # ---- COMPARISON TABLE: rows = family x H  (DSR_win = the paper's honest bar) ----
    print("\nCOMPARISON TABLE (per family x H)   [DSR = deflated Sharpe, Bailey & LopezdePrado 2014]")
    print("-" * 100)
    print(f"{'family':<8}{'H':>5}{'PF':>8}{'Shp_ann':>10}{'Shp/bar':>9}"
          f"{'%posWin':>9}{'turnovr':>9}{'DSR_win':>9}{'DSR_bar':>9}{'OOSbars':>9}")
    print("-" * 100)
    hdf_sorted = hdf.sort_values(["family", "H"], na_position="last").reset_index(drop=True)
    dwin_by_key = {r["key"]: r["_dsr_win"] for r in all_recs}
    dbar_by_key = {r["key"]: r["_dsr_bar"] for r in all_recs}
    for _, row in hdf_sorted.iterrows():
        dw = dwin_by_key.get(row["key"], np.nan)
        db = dbar_by_key.get(row["key"], np.nan)
        Hs = "-" if row["H"] is None or (isinstance(row["H"], float) and np.isnan(row["H"])) else f"{int(row['H'])}"
        turn = f"{row['turnover']:.3f}" if np.isfinite(row["turnover"]) else "n/a"
        print(f"{row['family']:<8}{Hs:>5}{row['pf']:>8.3f}{row['sr_ann']:>10.1f}"
              f"{row['sr_bar']:>9.4f}{row['pct_pos_win']*100:>8.0f}%{turn:>9}"
              f"{dw:>9.3f}{db:>9.3f}{int(row['n_combo_bars']):>9}")
    print("-" * 100)
    print("POOLED PER FAMILY")
    for fam, g in hdf.groupby("family"):
        mt = fam_meta[fam]
        med_pf = g.pf.replace([np.inf, -np.inf], np.nan).median()
        med_srann = g.sr_ann.median()
        med_pos = g.pct_pos_win.median()
        nsurv_win = int(np.nansum(fam_dsr_win[fam] > DSR_SURV))
        nsurv_bar = int(np.nansum(fam_dsr_bar[fam] > DSR_SURV))
        print(f"  {fam:<6} median PF {med_pf:.3f} | median Shp_ann {med_srann:.1f} | "
              f"median %pos {med_pos*100:.0f}% | DSR>{DSR_SURV} survivors: "
              f"window {nsurv_win}/{mt['n']}, bar {nsurv_bar}/{mt['n']}  "
              f"(N_trials={mt['N']}, SR0_win={mt['sr0_win']:.4f})")

    # ---- DSR survivor accounting (overall) ----
    nfin = int(hdf.sr_bar.notna().sum())
    surv_win_05 = sum(int(np.nansum(fam_dsr_win[f] > 0.5)) for f in fam_recs)
    surv_win_95 = sum(int(np.nansum(fam_dsr_win[f] > 0.95)) for f in fam_recs)
    surv_bar_05 = sum(int(np.nansum(fam_dsr_bar[f] > 0.5)) for f in fam_recs)
    surv_bar_95 = sum(int(np.nansum(fam_dsr_bar[f] > 0.95)) for f in fam_recs)
    # contrast: PSR>0.95 vs SR0=0 (no haircut), per-window
    momsw = [_sharpe_moments(_per_window_stream(r)) for r in all_recs]
    psr0 = psr(np.array([m["sr"] for m in momsw]), np.array([m["n"] for m in momsw]),
               np.array([m["skew"] for m in momsw]), np.array([m["kurt"] for m in momsw]), 0.0)
    surv_psr0 = int(np.nansum(psr0 > 0.95))

    print("\nDSR / DEFLATED SHARPE  (per-family benchmark; SR0 = expected best-of-N noise Sharpe)")
    print("-" * 100)
    print(f"  PRIMARY (per-WINDOW Sharpe, honest n=#windows): "
          f"DSR>0.5: {surv_win_05}/{n_combos_total}   DSR>0.95: {surv_win_95}/{n_combos_total}")
    print(f"  SENSITIVITY (per-BAR Sharpe, optimistic n=#bars): "
          f"DSR>0.5: {surv_bar_05}/{n_combos_total}   DSR>0.95: {surv_bar_95}/{n_combos_total}")
    print(f"  contrast: PSR>0.95 vs SR0=0 (NO multiple-testing haircut, per-window) "
          f"= {surv_psr0}/{nfin}")
    print("  note: per-window n is only ~#WFO windows so per-window DSR is conservative;")
    print("        per-bar n is inflated by rotation autocorrelation. Truth is between them.")
    print("        PF + IS->OOS rank-persistence are the robust anchors at this window count.")

    print("\nCROSS-VALIDATION OF SELECTION")
    print("-" * 100)
    print(f"  [PRIMARY at low window count] IS->OOS rank persistence over {rp['n_win']} windows:")
    print(f"     mean per-window Spearman rho = {rp['mean_rho']:+.3f}   "
          f"pooled rho = {rp['pooled_rho']:+.3f}   frac windows rho>0 = "
          f"{(rp['frac_pos'] or 0)*100:.0f}%")
    print(f"     (rho>0 => IS winners stay OOS winners; rho<0 => overfit selection)")
    print(f"  [SECONDARY, marginal] CSCV PBO: S={S_used} blocks x {wper} win/block, "
          f"{nsplit} splits  ->  PBO = {pbo:.3f}  (median logit lambda {medlam:+.3f})")
    print(f"     CAVEAT: only {len(wins)} WFO windows; CSCV is statistically thin here "
          f"(needs more windows for a stable PBO).")

    print("\nREDUNDANCY / EFFECTIVE-N  (per-BAR OOS streams, pairwise-complete, NOT zero-filled)")
    print("-" * 100)
    print(f"  combos = {red['n_combos']}  pairs scored = {red['n_pairs']}")
    print(f"  median pairwise |rho| = {red['median_abs_rho']:.3f}   "
          f"% pairs <0.5 = {red['pct_lt_05']*100:.0f}%")
    print(f"  effective-N (Meucci eigen) = {red['enb_meucci']:.2f}   "
          f"(entropy) = {red['enb_entropy']:.2f}  of {red['n_combos']} combos")
    print(f"  caveat: {red['caveat']}")

    # ---- verdict ----
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    _print_verdict(hdf, surv_win_95, surv_bar_95, n_combos_total, rp, pbo, red, fam_meta)


def _print_verdict(hdf, surv95_win, surv95_bar, n_combos, rp, pbo, red, fam_meta):
    def med(s):
        return s.replace([np.inf, -np.inf], np.nan).median()
    lines = []
    for fam, g in hdf.groupby("family"):
        Nf = fam_meta[fam]["N"]
        lines.append(f"- {fam:<5} XS: median OOS PF {med(g.pf):.3f}, median annualized Sharpe "
                     f"{g.sr_ann.median():.1f} (per-bar Sharpe {g.sr_bar.median():.4f}); "
                     f"{(g.pf>1).sum()}/{len(g)} horizons PF>1; DSR N_trials={Nf}.")
    lines.append(f"- DSR survivors (DSR>0.95): per-window {surv95_win}/{n_combos}, "
                 f"per-bar {surv95_bar}/{n_combos}.")
    rho = rp['pooled_rho'] or 0
    lines.append(f"- IS->OOS rank persistence pooled rho = {rp['pooled_rho']:+.3f} "
                 f"({'POSITIVE -> IS selection generalizes OOS' if rho > 0 else 'non-positive -> selection does NOT generalize'}); "
                 f"CSCV PBO = {pbo:.3f} (marginal, few windows).")
    lines.append(f"- Effective-N (per-bar, pairwise-complete) = {red['enb_meucci']:.2f} of "
                 f"{red['n_combos']} combos; median |rho| = {red['median_abs_rho']:.3f}.")
    lines.append("- vs single-series T_ML (pooled PBO 0.62, 0-2/25k cleared DSR, effective-N ~40 "
                 "of 25k, median PF ~1.0): judged here by the SAME de Prado gates.")
    for l in lines:
        print(l)


def main():
    args = sys.argv[1:]
    if args:
        run_dirs = args
    else:
        run_dirs = [DEFAULT_MAIN] + sorted(glob.glob(DEFAULT_MODAL_GLOB))
    print("Run dirs:")
    for d in run_dirs:
        print("  ", d)
    report_runs(run_dirs)


if __name__ == "__main__":
    main()

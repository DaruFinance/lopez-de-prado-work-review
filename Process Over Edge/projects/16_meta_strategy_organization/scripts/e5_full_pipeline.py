"""
e5_full_pipeline.py -- the whole assembly line, chained end to end, run on a
final never-touched out-of-time window, versus the lone backtester (Sisyphus).

The capstone's Experiment 1 compared the SELECTION RULE in isolation (pick the
best in-sample backtest, versus gate the candidate set with the Deflated Sharpe
Ratio). This culminating experiment instead chains EVERY station of the research
production line into ONE disclosed pipeline and runs that pipeline, as a single
system, on time it never saw during any tuning. The stations, in order:

  1. Data           : load real one-minute crypto base bars (read only).
  2. Information     : build dollar bars (sampled on traded notional, not the
     bars               clock), the bar that López de Prado prefers for
                        statistical regularity.
  3. Labels          : triple-barrier first-touch labels on full intrabar
                        high/low (no close-only shortcut), side from a causal
                        moving-average crossover.
  4. Uniqueness      : sample-uniqueness weights from label-span concurrency, so
                        overlapping (non-independent) labels do not get counted
                        as independent evidence when the model is trained.
  5. Model           : a bagged-tree secondary classifier predicting the
                        probability that the primary side's bet is profitable,
                        trained with purged, embargoed cross-validation and the
                        uniqueness weights.
  6. Meta-label gate : act only when that probability clears a threshold; the
                        meta-model can veto or size a bet but never flip its side.
  7. Bet sizing      : size each surviving bet by the model's probability.
  8. Allocation      : Hierarchical Risk Parity across the surviving instrument
                        sleeves (reusing the portfolio-construction allocator).
  9. Deploy gate     : deflate the assembled, allocated track record against the
                        disclosed number of configurations searched; deploy only
                        if the Deflated Sharpe Ratio clears the bar.

Every tuning decision (the knob grid scan, the per-instrument configuration
choice, the per-instrument deploy decision) is made strictly on the TUNING
region. The final out-of-time window is held out and scored once, untouched.
Costs are realistic and charged on every turnover. The pipeline's realized
out-of-time performance is compared against (a) the Sisyphus best-in-sample pick
deployed unit-size on the same window and (b) buy-and-hold of each instrument.

ONE process. BLAS threads pinned to one. Memory bounded: one instrument at a
time, dollar bars (about 20k per instrument), never a dense indicator matrix.

Reuses, read only, the program's existing station code:
  lib/bars.py                         (information-driven bars)
  03_meta_labeling/scripts/tbm.py     (triple-barrier, primary, features)
  15_..._bootstrapping/scripts/uniqueness.py  (concurrency, uniqueness weights)
  13_portfolio_construction/.../run_portfolio.py  (HRP allocator)
  lib/overfit.py                      (DSR, expected-max-Sharpe, purged CV)
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import sys, time, json, importlib.util, warnings, resource
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ROOT = PROJ.parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT))

import config as cfg                                              # noqa: E402
from lib import bars as B                                          # noqa: E402
from lib import overfit as O                                      # noqa: E402


def _load_sibling(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # so numba-cached kernels can import by name
    spec.loader.exec_module(mod)
    return mod


tbm = _load_sibling(ROOT / "projects/03_meta_labeling/scripts/tbm.py", "tbm")
uniq = _load_sibling(
    ROOT / "projects/15_sample_uniqueness_bootstrapping/scripts/uniqueness.py", "uniqueness")
p13 = _load_sibling(
    ROOT / "projects/13_portfolio_construction/scripts/run_portfolio.py", "p13_portfolio")

from sklearn.ensemble import BaggingClassifier                    # noqa: E402
from sklearn.tree import DecisionTreeClassifier                   # noqa: E402
from scipy import stats as ss                                     # noqa: E402

TAB = PROJ / "tables"; FIG = PROJ / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CRYPTO_DIR = cfg.CRYPTO_1M     # 1m base bars (<PAIR>_1m.parquet); see DATA.md
INSTRUMENTS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT"]

N_TARGET_BARS = 20000          # dollar bars per instrument
COST_BP = 7.0                  # per side, bp of notional (crypto taker, realistic)
OOS_FRAC = 0.20                # final never-touched out-of-time window (last 20%)

# IS-tunable knob grid -- the disclosed "trials". Structural shape is fixed
# (EMA crossover primary + triple barrier + bagged-tree meta); these numeric
# knobs are scanned on the TUNING region only.
FAST_SLOW = [(10, 30), (20, 60), (30, 90)]
PT_SL = [(1.0, 1.0), (1.5, 1.0), (2.0, 1.5)]
MAX_HOLD = [25, 50, 100]
VOL_SPAN = 50
PARAM_GRID = [(fs, ps, mh) for fs in FAST_SLOW for ps in PT_SL for mh in MAX_HOLD]
N_TRIALS = len(PARAM_GRID)     # 27 configurations searched per instrument

N_SPLITS = 6
EMBARGO = 0.01
META_THRESH = 0.50
DSR_GATE = 0.95                # deploy bar (program standard)
MIN_EVENTS = 150


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# --------------------------------------------------------------------------- #
# Station helpers
# --------------------------------------------------------------------------- #
def make_dollar_bars(path: str) -> pd.DataFrame:
    base = tbm.load_base_crypto(path)
    return B.matched_bars(base, N_TARGET_BARS)["dollar"]


def bars_per_year(bars: pd.DataFrame) -> float:
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def trade_returns_to_series(ev_idx, hold, pnl, n_bars):
    """Spread each trade's net return over the bars it was held -> per-bar stream."""
    r = np.zeros(n_bars)
    for k in range(len(ev_idx)):
        i0 = int(ev_idx[k]); h = max(1, int(hold[k]))
        j1 = min(i0 + h, n_bars)
        r[i0:j1] += pnl[k] / h
    return r


def score(r: np.ndarray, bpy: float) -> dict:
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    nz = r[r != 0.0]
    sr = O.sharpe(r)
    gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
    pf = float(gp / gn) if gn > 0 else (np.nan if gp == 0 else np.inf)
    return dict(sr_ann=float(sr * np.sqrt(bpy)), sr_perbar=float(sr), pf=pf,
                skew=float(ss.skew(r)) if len(r) > 2 else 0.0,
                kurt=float(ss.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0,
                n_obs=int(len(r)))


# --------------------------------------------------------------------------- #
# Build one configuration on a bar range and return its per-bar net stream on a
# SUBSET of bars (the tuning region or the out-of-time region), strictly causal.
# --------------------------------------------------------------------------- #
def run_config(close, high, low, openp, bars_full, fast, slow, pt, sl, mh,
               cost, split_idx, want_model_for_oos=False):
    """Causal pipeline for one knob set.

    Trains the meta-model only on TUNING-region events (events whose entry bar is
    < split_idx). Produces:
      tune  : per-bar net stream over [0, split_idx)  (model OOF on tuning)
      oos   : per-bar net stream over [split_idx, n)  (model applied forward)
    plus the tuning-region label/uniqueness/model bookkeeping for the pass-through
    table. The split is on BAR index, so the out-of-time window is a clean
    forward block the tuning never saw.
    """
    n = len(close)
    vol = tbm.ewma_vol(close, VOL_SPAN)
    side_full = tbm.primary_ma_crossover(close, fast, slow)
    ev_all = tbm.crossover_events(side_full)
    warm = slow + 12
    ev_all = ev_all[(ev_all > warm) & (ev_all < n - 1)]
    if len(ev_all) < MIN_EVENTS:
        return None
    side_all = side_full[ev_all]
    tb = tbm.triple_barrier(close, high, low, openp, ev_all, side_all, vol, pt, sl, mh)
    ret_gross = tb["ret_gross"].to_numpy()
    hold = tb["hold"].to_numpy()
    touch = tb["touch"].to_numpy()
    pnl_net = ret_gross - 2.0 * cost
    y = (pnl_net > 0).astype(int)

    # ---- station 4: sample-uniqueness weights from label-span concurrency ----
    c = uniq.concurrency(ev_all, touch, n)
    u = uniq.avg_uniqueness(ev_all, touch, c)           # in [0,1]
    u = np.where(np.isfinite(u) & (u > 0), u, 1e-6)

    feat = tbm.build_features(bars_full, side_full, vol, fast, slow)
    X = np.nan_to_num(feat.iloc[ev_all].to_numpy(np.float64), 0.0, 0.0, 0.0)

    # split events into tuning vs out-of-time by entry bar
    is_tune = ev_all < split_idx
    tune_pos = np.where(is_tune)[0]
    oos_pos = np.where(~is_tune)[0]
    if len(tune_pos) < MIN_EVENTS or len(oos_pos) < 30:
        return None
    if y[tune_pos].sum() < 15 or (1 - y[tune_pos]).sum() < 15:
        return None

    # ---- station 5: meta-model with purged CV, weighted by uniqueness ----
    # Tuning OOF probabilities (leakage-free) drive the tuning-region stream and
    # the configuration selection. A model fit on ALL tuning events is then
    # applied forward to the out-of-time events (those were never in training).
    y_t = y[tune_pos]; X_t = X[tune_pos]; u_t = u[tune_pos]
    p_oof = np.full(len(tune_pos), float(y_t.mean()))
    for tr, te in O.purged_kfold_splits(len(tune_pos), N_SPLITS, EMBARGO, label_span=3):
        if len(tr) < 50 or y_t[tr].sum() < 5 or (1 - y_t[tr]).sum() < 5:
            continue
        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
            n_estimators=40, max_samples=0.8, max_features=0.8,
            bootstrap=True, n_jobs=1, random_state=0)
        clf.fit(X_t[tr], y_t[tr], sample_weight=u_t[tr])
        if len(clf.classes_) == 1:
            p_oof[te] = float(clf.classes_[0])
        else:
            pi = list(clf.classes_).index(1)
            p_oof[te] = clf.predict_proba(X_t[te])[:, pi]
    p_oof = np.nan_to_num(p_oof, nan=float(y_t.mean()))

    # tuning-region meta stream (sized by prob, gated by threshold, costed)
    act_t = p_oof >= META_THRESH
    size_t = np.where(act_t, p_oof, 0.0)
    pnl_meta_t = size_t * ret_gross[tune_pos] - 2.0 * cost * size_t
    r_tune = trade_returns_to_series(ev_all[tune_pos], hold[tune_pos], pnl_meta_t, n)
    r_tune = r_tune[:split_idx]

    out = dict(
        n_ev=len(ev_all), n_ev_tune=len(tune_pos), n_ev_oos=len(oos_pos),
        base_rate=float(y.mean()),
        avg_uniqueness=float(np.mean(u)),
        eff_sample=float(np.sum(u)), naive_n=int(len(u)),
        frac_act_tune=float(act_t.mean()),
        r_tune=r_tune)

    if want_model_for_oos:
        # fit on ALL tuning events (uniqueness-weighted), apply forward to OOS
        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20),
            n_estimators=40, max_samples=0.8, max_features=0.8,
            bootstrap=True, n_jobs=1, random_state=0)
        clf.fit(X_t, y_t, sample_weight=u_t)
        X_o = X[oos_pos]
        if len(clf.classes_) == 1:
            p_oos = np.full(len(oos_pos), float(clf.classes_[0]))
        else:
            pi = list(clf.classes_).index(1)
            p_oos = clf.predict_proba(X_o)[:, pi]
        # meta sleeve on the out-of-time window
        act_o = p_oos >= META_THRESH
        size_o = np.where(act_o, p_oos, 0.0)
        pnl_meta_o = size_o * ret_gross[oos_pos] - 2.0 * cost * size_o
        r_oos_meta = trade_returns_to_series(
            ev_all[oos_pos] - split_idx, hold[oos_pos], pnl_meta_o, n - split_idx)
        # primary-only (Sisyphus) sleeve on the out-of-time window: act on every
        # event at unit size, same costs
        pnl_prim_o = ret_gross[oos_pos] - 2.0 * cost
        r_oos_prim = trade_returns_to_series(
            ev_all[oos_pos] - split_idx, hold[oos_pos], pnl_prim_o, n - split_idx)
        out.update(r_oos_meta=r_oos_meta, r_oos_prim=r_oos_prim,
                   frac_act_oos=float(act_o.mean()), n_act_oos=int(act_o.sum()))
    return out


# --------------------------------------------------------------------------- #
# Per-instrument: scan the grid on the tuning region, pick configs two ways.
# --------------------------------------------------------------------------- #
def run_instrument(name: str) -> dict | None:
    path = os.path.join(CRYPTO_DIR, f"{name}_1m.parquet")
    if not os.path.exists(path):
        return None
    bars = make_dollar_bars(path)
    if len(bars) < 4000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    split_idx = int(round(n * (1.0 - OOS_FRAC)))
    cost = COST_BP / 1e4

    # ---- scan the disclosed grid on the TUNING region only ----
    trials = []
    for (fast, slow), (pt, sl), mh in PARAM_GRID:
        r = run_config(close, high, low, openp, bars, fast, slow, pt, sl, mh,
                       cost, split_idx, want_model_for_oos=False)
        if r is None:
            continue
        st = score(r["r_tune"], bpy)
        r.update(fast=fast, slow=slow, pt=pt, sl=sl, mh=mh,
                 tune_sr_perbar=st["sr_perbar"], tune_sr_ann=st["sr_ann"],
                 tune_pf=st["pf"], tune_skew=st["skew"], tune_kurt=st["kurt"],
                 tune_nobs=st["n_obs"])
        trials.append(r)
    if not trials:
        return None

    sr_trials = np.array([t["tune_sr_perbar"] for t in trials])
    # disclosed deflation: deflate the BEST tuning config against the dispersion
    # and count of the configurations actually searched (the trial set)
    best = int(np.argmax(sr_trials))
    T = trials[best]
    dd = O.deflated_sharpe_ratio(T["tune_sr_perbar"], T["tune_nobs"],
                                 T["tune_skew"], T["tune_kurt"], sr_trials)

    # the assembly line will only deploy this instrument's sleeve if the tuning
    # DSR clears the gate; either way we score the same selected config forward
    Tcfg = (T["fast"], T["slow"]), (T["pt"], T["sl"]), T["mh"]
    (af, asw), (apt, asl), amh = Tcfg
    fwd = run_config(close, high, low, openp, bars, af, asw, apt, asl, amh,
                     cost, split_idx, want_model_for_oos=True)
    if fwd is None:
        return None

    # buy-and-hold on the out-of-time window (per-bar log returns)
    bh = np.diff(np.log(close[split_idx - 1:]))
    bh = bh[:len(fwd["r_oos_meta"])]

    s_meta = score(fwd["r_oos_meta"], bpy)
    s_prim = score(fwd["r_oos_prim"], bpy)
    s_bh = score(bh, bpy)

    return dict(
        name=name, n_bars=n, split_idx=split_idx, bpy=bpy,
        n_configs=len(trials),
        best_cfg=f"f{T['fast']}/s{T['slow']} pt{T['pt']}/sl{T['sl']} h{T['mh']}",
        # station pass-through (on the selected config)
        n_ev=fwd["n_ev"], n_ev_tune=fwd["n_ev_tune"], n_ev_oos=fwd["n_ev_oos"],
        base_rate=fwd["base_rate"], avg_uniqueness=fwd["avg_uniqueness"],
        eff_sample=fwd["eff_sample"], naive_n=fwd["naive_n"],
        frac_act_oos=fwd["frac_act_oos"], n_act_oos=fwd["n_act_oos"],
        # tuning selection + deploy gate
        tune_sr_ann=T["tune_sr_ann"], tune_pf=T["tune_pf"],
        tune_dsr=float(dd["dsr"]) if np.isfinite(dd["dsr"]) else 0.0,
        tune_sr0=float(dd["sr0"]),
        deploy=bool(np.isfinite(dd["dsr"]) and dd["dsr"] >= DSR_GATE),
        # realized out-of-time
        pipe_oos_sr_ann=s_meta["sr_ann"], pipe_oos_pf=s_meta["pf"],
        sis_oos_sr_ann=s_prim["sr_ann"], sis_oos_pf=s_prim["pf"],
        bh_oos_sr_ann=s_bh["sr_ann"], bh_oos_pf=s_bh["pf"],
        # carry per-bar OOS streams for HRP + pooled scoring + figure
        _r_oos_meta=fwd["r_oos_meta"], _r_oos_prim=fwd["r_oos_prim"], _bh=bh)


# --------------------------------------------------------------------------- #
# Allocation + pooled deploy decision
# --------------------------------------------------------------------------- #
def allocate_and_score(results):
    """Station 8 + 9: HRP across the DEPLOYED instrument sleeves on a common OOS
    bar index, then deflate the assembled track record against the disclosed
    program-level trial count (configs searched across all instruments)."""
    deployed = [r for r in results if r["deploy"]]
    bpy = np.mean([r["bpy"] for r in results])

    def align(streams):
        L = min(len(s) for s in streams)
        return np.column_stack([s[:L] for s in streams])

    # baseline length = the common out-of-time window across all instruments
    oos_len = min(len(r["_r_oos_prim"]) for r in results)

    # assembled pipeline: HRP weights over DEPLOYED sleeves (cash if none deploy)
    if len(deployed) == 0:
        pipe_pool = np.zeros(oos_len); w = np.array([]); n_dep = 0
    elif len(deployed) == 1:
        pipe_pool = deployed[0]["_r_oos_meta"].copy(); w = np.array([1.0]); n_dep = 1
    else:
        M = align([r["_r_oos_meta"] for r in deployed])
        cov = np.atleast_2d(np.cov(M.T))
        try:
            w = p13.w_hrp(cov, use_numba=True)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                raise ValueError
            w = w / w.sum()
        except Exception:
            w = np.full(M.shape[1], 1.0 / M.shape[1])
        pipe_pool = M @ w; n_dep = len(deployed)

    # baselines pooled equal-weight across ALL instruments (a desk with no gate)
    Msis = align([r["_r_oos_prim"] for r in results])
    Mbh = align([r["_bh"] for r in results])
    sis_pool = Msis.mean(1)
    bh_pool = Mbh.mean(1)

    def sc(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        nz = x[x != 0]
        gp = nz[nz > 0].sum(); gn = -nz[nz < 0].sum()
        pf = float(gp / gn) if gn > 0 else np.nan
        return float(O.sharpe(x) * np.sqrt(bpy)), pf, x

    pipe_sr, pipe_pf, pipe_x = sc(pipe_pool)
    sis_sr, sis_pf, sis_x = sc(sis_pool)
    bh_sr, bh_pf, bh_x = sc(bh_pool)

    # program-level deflation of the assembled pipeline's realized OOS track
    n_program_trials = sum(r["n_configs"] for r in results)
    sr_trials_proxy = np.array([r["tune_sr_ann"] / np.sqrt(bpy) for r in results
                                for _ in range(r["n_configs"] // len(results) + 1)])
    if len(pipe_x) > 2 and pipe_x.std(ddof=1) > 0:
        pipe_dsr = O.deflated_sharpe_ratio(
            O.sharpe(pipe_x), len(pipe_x), float(ss.skew(pipe_x)),
            float(ss.kurtosis(pipe_x, fisher=False)), sr_trials_proxy)
        pipe_dsr_v = float(pipe_dsr["dsr"]) if np.isfinite(pipe_dsr["dsr"]) else 0.0
    else:
        pipe_dsr_v = 0.0

    return dict(
        n_deployed=n_dep, hrp_weights=w.tolist(),
        deployed_names=[r["name"] for r in deployed],
        n_program_trials=int(n_program_trials),
        pipe_oos_sr_ann=pipe_sr, pipe_oos_pf=pipe_pf, pipe_oos_dsr=pipe_dsr_v,
        sis_oos_sr_ann=sis_sr, sis_oos_pf=sis_pf,
        bh_oos_sr_ann=bh_sr, bh_oos_pf=bh_pf,
        deploys=bool(pipe_dsr_v >= DSR_GATE and n_dep > 0),
        _pipe_x=pipe_x, _sis_x=sis_x, _bh_x=bh_x)


# --------------------------------------------------------------------------- #
# Tables + figure
# --------------------------------------------------------------------------- #
def write_tables(results, pool):
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    df = pd.DataFrame(rows).sort_values("name")
    # round
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(TAB / "full_pipeline.csv", index=False)

    # headline scoreboard
    head = {
        "n_instruments": len(results),
        "n_configs_per_instrument": N_TRIALS,
        "n_program_configs_searched": pool["n_program_trials"],
        "n_sleeves_deployed": pool["n_deployed"],
        "deployed_instruments": pool["deployed_names"],
        "hrp_weights": [round(x, 4) for x in pool["hrp_weights"]],
        "pipeline_oos_sharpe_ann": round(pool["pipe_oos_sr_ann"], 4),
        "pipeline_oos_pf": round(pool["pipe_oos_pf"], 4) if np.isfinite(pool["pipe_oos_pf"]) else None,
        "pipeline_oos_dsr": round(pool["pipe_oos_dsr"], 4),
        "pipeline_clears_dsr_gate": pool["deploys"],
        "sisyphus_oos_sharpe_ann": round(pool["sis_oos_sr_ann"], 4),
        "sisyphus_oos_pf": round(pool["sis_oos_pf"], 4) if np.isfinite(pool["sis_oos_pf"]) else None,
        "buyhold_oos_sharpe_ann": round(pool["bh_oos_sr_ann"], 4),
        "buyhold_oos_pf": round(pool["bh_oos_pf"], 4) if np.isfinite(pool["bh_oos_pf"]) else None,
        "dsr_gate": DSR_GATE, "oos_fraction": OOS_FRAC, "cost_bp_per_side": COST_BP,
    }
    (TAB / "full_pipeline_headline.json").write_text(json.dumps(head, indent=2, default=float))

    # markdown
    md = []
    md.append("# Full assembly-line pipeline on held-out time\n")
    md.append("_One disclosed system: dollar bars -> triple-barrier labels -> "
              "uniqueness weights -> bagged-tree meta-model -> meta-label gate -> "
              "bet sizing -> Hierarchical Risk Parity -> Deflated-Sharpe deploy gate. "
              "The final out-of-time window is held out and scored once._\n")
    md.append("\n## Headline (pooled out-of-time window)\n")
    md.append("| process | OOS Sharpe (ann) | OOS PF | OOS DSR | deploys? |")
    md.append("|---|---|---|---|---|")
    pf_pipe = f"{pool['pipe_oos_pf']:.3f}" if np.isfinite(pool['pipe_oos_pf']) else "n/a"
    pf_sis = f"{pool['sis_oos_pf']:.3f}" if np.isfinite(pool['sis_oos_pf']) else "n/a"
    pf_bh = f"{pool['bh_oos_pf']:.3f}" if np.isfinite(pool['bh_oos_pf']) else "n/a"
    md.append(f"| assembled pipeline (DSR-gated) | {pool['pipe_oos_sr_ann']:.3f} | "
              f"{pf_pipe} | {pool['pipe_oos_dsr']:.3f} | "
              f"{'YES' if pool['deploys'] else 'NO'} |")
    md.append(f"| Sisyphus best-in-sample pick | {pool['sis_oos_sr_ann']:.3f} | "
              f"{pf_sis} | (no deflation) | deploys blindly |")
    md.append(f"| buy and hold | {pool['bh_oos_sr_ann']:.3f} | {pf_bh} | - | - |")
    md.append(f"\nProgram disclosed {pool['n_program_trials']} configurations searched "
              f"across {len(results)} instruments. Sleeves deployed by the gate: "
              f"{pool['n_deployed']} ({', '.join(pool['deployed_names']) or 'none'}).\n")
    md.append("\n## Per-instrument station pass-through (selected configuration)\n")
    cols = ["name", "n_bars", "n_ev", "n_ev_tune", "n_ev_oos", "base_rate",
            "avg_uniqueness", "eff_sample", "naive_n", "frac_act_oos",
            "tune_sr_ann", "tune_dsr", "deploy",
            "pipe_oos_sr_ann", "sis_oos_sr_ann", "bh_oos_sr_ann", "best_cfg"]
    md.append(df[cols].to_markdown(index=False))
    md.append("\n\n_Stations: n_ev = triple-barrier labeled events; "
              "avg_uniqueness and eff_sample (= sum of uniqueness weights, versus "
              "naive_n events) are the sample-uniqueness station; frac_act_oos is "
              "the meta-label gate's act rate out of sample; tune_dsr is the "
              "per-instrument deploy gate; the three OOS Sharpe columns are the "
              "realized held-out result of the pipeline sleeve, the Sisyphus "
              "primary-only sleeve, and buy-and-hold._\n")
    (TAB / "full_pipeline.md").write_text("\n".join(md))
    print(f"wrote {TAB/'full_pipeline.csv'}")
    print(f"wrote {TAB/'full_pipeline.md'}")
    print(f"wrote {TAB/'full_pipeline_headline.json'}")
    return df, head


def make_figure(results, pool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from lib import style
    style.set_style()
    P = style.PALETTE

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.32)

    # --- top: the station chain ---
    axc = fig.add_subplot(gs[0]); axc.axis("off")
    axc.set_xlim(0, 10); axc.set_ylim(0, 3)
    stations = ["1. Data\n(1-min base)", "2. Dollar\nbars", "3. Triple-\nbarrier labels",
                "4. Uniqueness\nweights", "5. Tree\nmeta-model", "6. Meta-label\ngate",
                "7. Bet\nsizing", "8. HRP\nallocation", "9. DSR\ndeploy gate"]
    cols = ["#009E73", "#009E73", "#E69F00", "#E69F00", "#56B4E9",
            "#56B4E9", "#CC79A7", "#0072B2", "#D55E00"]
    bw = 0.92; gap = (10 - len(stations) * bw) / (len(stations) + 1)
    cx = []
    for i, (s, col) in enumerate(zip(stations, cols)):
        x = gap + i * (bw + gap)
        axc.add_patch(FancyBboxPatch((x, 1.15), bw, 1.05,
                      boxstyle="round,pad=0.02,rounding_size=0.06",
                      linewidth=1.5, edgecolor=col, facecolor=col + "22"))
        axc.text(x + bw / 2, 1.67, s, ha="center", va="center", fontsize=7.4,
                 fontweight="bold", color="#222222")
        cx.append(x + bw / 2)
    for i in range(len(stations) - 1):
        axc.add_patch(FancyArrowPatch((cx[i] + bw / 2, 1.67), (cx[i + 1] - bw / 2, 1.67),
                      arrowstyle="-|>", mutation_scale=11, lw=1.4, color="#888888"))
    axc.text(5, 2.72, "The research assembly line, chained as one disclosed system and "
             "run on held-out time", ha="center", va="center", fontsize=13,
             fontweight="bold")
    dep = ", ".join(pool["deployed_names"]) or "none"
    axc.text(5, 0.55, f"Disclosed configurations searched: {pool['n_program_trials']}   |   "
             f"sleeves cleared to deploy: {pool['n_deployed']} ({dep})   |   "
             f"pipeline deploys: {'YES' if pool['deploys'] else 'NO'}",
             ha="center", va="center", fontsize=9.2, color="#444444")

    # --- bottom: equity comparison on the out-of-time window ---
    axe = fig.add_subplot(gs[1])
    px = pool["_pipe_x"]; sx = pool["_sis_x"]; bx = pool["_bh_x"]
    L = min(len(px), len(sx), len(bx))
    axe.plot(np.cumsum(sx[:L]), color="#999999", lw=1.6,
             label=f"Sisyphus best-in-sample pick (Sharpe {pool['sis_oos_sr_ann']:.2f})")
    axe.plot(np.cumsum(bx[:L]), color=P["volume"], lw=1.6,
             label=f"buy and hold (Sharpe {pool['bh_oos_sr_ann']:.2f})")
    pipe_lab = (f"assembled pipeline, DSR-gated (Sharpe {pool['pipe_oos_sr_ann']:.2f}, "
                f"DSR {pool['pipe_oos_dsr']:.2f})" if pool["n_deployed"] > 0
                else "assembled pipeline: nothing cleared the gate -> cash")
    axe.plot(np.cumsum(px[:L]), color=P["dollar"], lw=2.2, label=pipe_lab)
    axe.axhline(0, color="k", lw=0.7, ls="--")
    axe.set_xlabel("out-of-time bar (held-out window, never used in tuning)")
    axe.set_ylabel("cumulative net return (per-bar, risk units)")
    axe.set_title("Realized out-of-time performance: assembled pipeline vs lone "
                  "backtester vs buy and hold")
    axe.legend(loc="best", fontsize=9.5)
    fig.savefig(FIG / "fig_full_pipeline.png")
    fig.savefig(FIG / "fig_full_pipeline.svg")
    plt.close(fig)
    print(f"wrote {FIG/'fig_full_pipeline.png'} (+ .svg)")


# --------------------------------------------------------------------------- #
def main():
    print("=== E5: the full assembly-line pipeline on held-out time ===", flush=True)
    print(f"instruments: {INSTRUMENTS}", flush=True)
    print(f"{N_TRIALS} configurations searched per instrument; final {OOS_FRAC:.0%} "
          f"of bars held out; cost {COST_BP} bp/side; DSR gate {DSR_GATE}", flush=True)
    t0 = time.time()
    results = []
    for nm in INSTRUMENTS:
        tt = time.time()
        try:
            r = run_instrument(nm)
        except Exception as e:
            print(f"  [ERR] {nm}: {e}", flush=True)
            continue
        if r is None:
            print(f"  [skip] {nm} (insufficient data/events)", flush=True)
            continue
        results.append(r)
        print(f"  [ok] {nm:9s} tuneDSR={r['tune_dsr']:.3f} deploy={r['deploy']!s:5s} "
              f"pipeOOS_SR={r['pipe_oos_sr_ann']:+.2f} sisOOS_SR={r['sis_oos_sr_ann']:+.2f} "
              f"bhOOS_SR={r['bh_oos_sr_ann']:+.2f} uniq={r['avg_uniqueness']:.3f} "
              f"({time.time()-tt:.0f}s, rss={_rss_gb():.1f}G)", flush=True)
    if not results:
        print("no instruments produced results"); return
    pool = allocate_and_score(results)
    df, head = write_tables(results, pool)
    make_figure(results, pool)
    print(f"\n--- HEADLINE ---")
    for k, v in head.items():
        print(f"  {k}: {v}")
    print(f"\nTOTAL {time.time()-t0:.0f}s  peak RSS {_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()

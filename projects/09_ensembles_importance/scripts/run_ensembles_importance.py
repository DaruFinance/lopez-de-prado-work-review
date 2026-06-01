#!/usr/bin/env python3
"""
run_ensembles_importance.py, Ensembles (bagging vs boosting + hyper-tuning) and
feature importance (MDI vs MDA vs clustered-MDA) on a labeled ML task.
(López de Prado, AFML Ch.6 ensembles, Ch.8 importance, Ch.9 hyper-tuning;
 ML4AM Ch.6 feature importance / clustered MDA.)

This is a research driver. On a triple-barrier-labeled, causal-feature ML task,
per instrument, across CRYPTO + US EQUITIES + FOREX, costed and PURGED, it answers
two linked LdP questions:

(1) BAGGING vs BOOSTING
    A bagged tree ensemble (RandomForest-style: low max_features, a positive
    min_weight_fraction_leaf, sample-weighted by label uniqueness, and sequential-
    bootstrap max_samples = average uniqueness, AFML 4.5/6.2/6.3) versus a boosted
    model (HistGradientBoosting). Hyper-tuning is scored by NEGATIVE LOG-LOSS over a
    PURGED grid search (AFML 9.4: do NOT tune by accuracy). Control = the same tuning
    scored by ACCURACY. We compare OUT-OF-SAMPLE Deflated Sharpe of the resulting
    bet, and the IS-OOS overfit gap (log-loss train minus OOS).

(2) FEATURE IMPORTANCE
    MDI (in-sample, tree impurity) vs MDA (permutation, out-of-fold, log-loss
    scored) vs CLUSTERED-MDA (cluster correlated features, permute whole clusters,
    AFML 8.5 / ML4AM 6). We show (a) MDI's known substitution bias toward
    high-cardinality / correlated features, and (b) the STABILITY of the top-feature
    set across CPCV paths (AFML Ch.12) for each method.

HEADLINE METRIC = Deflated Sharpe Ratio (lib/overfit.deflated_sharpe_ratio), net of
realistic per-turnover costs. PURGED CV / CPCV from lib/overfit with label_span set
to the label horizon. Triple-barrier labels + causal features reused from
projects/03_meta_labeling/scripts/tbm.py.

Numba'd (NON-sklearn hot loops only; sklearn fits are left to sklearn):
  - sequential bootstrap draw (AFML 4.5.2)        -> _seqboot_kernel
  - co-event count + average label uniqueness     -> _avg_uniqueness_kernel
Each is verified BIT-IDENTICAL against an independent pure-Python reference in
--verify (and at import-time in --smoke).

Run:
  python3 scripts/run_ensembles_importance.py --smoke    # tiny 1-core sanity (sec)
  python3 scripts/run_ensembles_importance.py --profile  # cProfile 1 instrument
  python3 scripts/run_ensembles_importance.py --verify   # numba == python checks
  python3 scripts/run_ensembles_importance.py            # FULL multi-market run
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
REPO_ROOT = ROOT = _d
sys.path.insert(0, REPO_ROOT)
import config as cfg
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
# reuse the meta-labeling tbm.py (triple barrier + causal features)
sys.path.insert(0, os.path.join(ROOT, "projects", "03_meta_labeling", "scripts"))

from lib import bars as B
from lib import overfit as O
from lib import realism as RZ
import tbm

from scipy import stats as ss
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CRYPTO_DIR = cfg.CRYPTO_1M
FX_DIR = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

# per-side cost in bp of notional (entry AND exit). Costed P&L only.
COST_BP = {"crypto": 7.0, "equities": 2.0, "forex": 1.0}
N_TARGET_BARS = 20000          # dollar/tick bars per instrument

# Fixed STRUCTURAL task (one labeling config; this study is NOT a knob sweep over
# the label, it is a model/importance comparison ON a fixed labeled task).
FAST, SLOW = 20, 60
PT_MULT, SL_MULT = 1.5, 1.0
MAX_HOLD = 50
VOL_SPAN = 50

N_SPLITS = 6                   # purged k-fold for OOF predictions
EMBARGO = 0.01
CPCV_GROUPS = 6                # CPCV paths for importance-stability (C(6,2)=15)
CPCV_K = 2
ACT_THRESH = 0.50              # act on a bet when P(profit) >= this; size ~ p

# Hyper-parameter grids (the "trials" for DSR multiple-testing). Small & sensible:
# the model comparison, not a grid blowout, is the point. (sizes set after profiling)
RF_GRID = {
    "max_features": [1, 2, 3],            # LOW max_features (AFML 6.2 decorrelation)
    "min_weight_fraction_leaf": [0.0, 0.05],
    "n_estimators": [200],
}
HGB_GRID = {
    "max_iter": [150, 300],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
}
RANDOM_STATE = 0


# =========================================================================== #
# NUMBA HOT LOOPS  (non-sklearn), sequential bootstrap & average uniqueness
# =========================================================================== #
@njit(cache=True)
def _avg_uniqueness_kernel(t0, t1, n_bars):
    """Average label uniqueness per event (AFML 4.4).

    Each event k spans bars [t0[k], t1[k]] (inclusive). c[b] = # events live at
    bar b. The uniqueness of event k = mean over its bars of 1/c[b]. Two passes:
    first accumulate concurrency counts, then average the reciprocal over span."""
    n_ev = t0.shape[0]
    c = np.zeros(n_bars, np.int64)
    for k in range(n_ev):
        for b in range(t0[k], t1[k] + 1):
            c[b] += 1
    u = np.empty(n_ev, np.float64)
    for k in range(n_ev):
        s = 0.0
        m = 0
        for b in range(t0[k], t1[k] + 1):
            s += 1.0 / c[b]
            m += 1
        u[k] = s / m if m > 0 else 0.0
    return u, c


@njit(cache=True)
def _seqboot_kernel(t0, t1, n_bars, n_draws, rand_u):
    """Sequential bootstrap draw (AFML 4.5.2).

    Draw `n_draws` event indices one at a time. Before each draw, the sampling
    probability of candidate k is proportional to its average uniqueness GIVEN
    the bars already occupied by previously-drawn events (reduces overlapping
    redundancy vs. standard bootstrap). `rand_u` is a precomputed U(0,1) stream
    (one per draw) so the kernel is deterministic given the RNG state, this is
    what makes the numba/python parity check exact."""
    n_ev = t0.shape[0]
    occ = np.zeros(n_bars, np.int64)          # times each bar already drawn
    out = np.empty(n_draws, np.int64)
    prob = np.empty(n_ev, np.float64)
    for d in range(n_draws):
        tot = 0.0
        for k in range(n_ev):
            s = 0.0
            m = 0
            for b in range(t0[k], t1[k] + 1):
                s += 1.0 / (occ[b] + 1.0)     # uniqueness given current occupancy
                m += 1
            uk = s / m if m > 0 else 0.0
            prob[k] = uk
            tot += uk
        # inverse-CDF sample from the (normalised) prob vector
        target = rand_u[d] * tot
        cum = 0.0
        pick = n_ev - 1
        for k in range(n_ev):
            cum += prob[k]
            if cum >= target:
                pick = k
                break
        out[d] = pick
        for b in range(t0[pick], t1[pick] + 1):
            occ[b] += 1
    return out


# --- independent pure-python references (for bit-identical verification) --- #
def _avg_uniqueness_ref(t0, t1, n_bars):
    n_ev = len(t0)
    c = np.zeros(n_bars, np.int64)
    for k in range(n_ev):
        c[t0[k]:t1[k] + 1] += 1
    u = np.empty(n_ev, np.float64)
    for k in range(n_ev):
        seg = c[t0[k]:t1[k] + 1]
        u[k] = (1.0 / seg).mean() if len(seg) else 0.0
    return u, c


def _seqboot_ref(t0, t1, n_bars, n_draws, rand_u):
    n_ev = len(t0)
    occ = np.zeros(n_bars, np.int64)
    out = np.empty(n_draws, np.int64)
    for d in range(n_draws):
        prob = np.empty(n_ev)
        for k in range(n_ev):
            seg = occ[t0[k]:t1[k] + 1]
            prob[k] = (1.0 / (seg + 1.0)).mean() if len(seg) else 0.0
        tot = prob.sum()
        target = rand_u[d] * tot
        cum = 0.0
        pick = n_ev - 1
        for k in range(n_ev):
            cum += prob[k]
            if cum >= target:
                pick = k
                break
        out[d] = pick
        occ[t0[pick]:t1[pick] + 1] += 1
    return out


def avg_uniqueness(t0, t1, n_bars):
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    return _avg_uniqueness_kernel(t0, t1, int(n_bars))


def seq_bootstrap(t0, t1, n_bars, n_draws, rng):
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    rand_u = rng.random(int(n_draws))
    return _seqboot_kernel(t0, t1, int(n_bars), int(n_draws), rand_u)


# =========================================================================== #
# Data / labeled-task construction (reusing tbm)
# =========================================================================== #
def make_bars(market: str, path: str) -> pd.DataFrame:
    if market == "crypto":
        base = tbm.load_base_crypto(path)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "equities":
        base = B.load_base_equity_etf(path, rth=True)
        return B.matched_bars(base, N_TARGET_BARS)["dollar"]
    if market == "forex":
        base = tbm.load_base_fx(path)
        thr = base["count"].sum() / N_TARGET_BARS
        return B.threshold_bars(base, "count", thr)
    raise ValueError(market)


def bars_per_year(bars: pd.DataFrame) -> float:
    idx = bars.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    span_s = (idx.view("int64")[-1] - idx.view("int64")[0]) / 1e9
    years = max(span_s / (365.25 * 24 * 3600), 1e-6)
    return len(bars) / years


def build_task(market, path):
    """Return (bars, X, y, ev, side, ret_gross, hold, t0, t1, cost, bpy, feat_names).

    A fixed triple-barrier labeled task: y = 1[net P&L of the structural bet > 0].
    t0/t1 are the bar span of each event's label (for purging + uniqueness)."""
    bars = make_bars(market, path)
    if len(bars) < 4000:
        return None
    bpy = bars_per_year(bars)
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n = len(close)
    # REALISTIC per-side cost (time-of-day half-spread + commission) per bar for
    # EQUITY/FOREX; crypto keeps the flat house default. Causal (ex-ante).
    sym = (os.path.basename(path).replace("_fx1m.parquet", "")
           .replace("_1m.parquet", "").replace(".csv.gz", ""))
    cost_side = RZ.per_side_cost_fraction(market, sym, bars.index, close,
                                          crypto_fallback=COST_BP[market] / 1e4)

    vol = tbm.ewma_vol(close, VOL_SPAN)
    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    ev = tbm.crossover_events(side_full)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]
    if len(ev) < 300:
        return None
    side = side_full[ev]
    tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol,
                            PT_MULT, SL_MULT, MAX_HOLD)
    ret_gross = tb["ret_gross"].to_numpy()
    hold = tb["hold"].to_numpy()
    # per-trade realistic round-trip cost = entry-bar + exit-bar per-side cost
    ex_bar = np.minimum(ev + np.maximum(1, hold.astype(np.int64)), n - 1)
    rt_cost = cost_side[ev] + cost_side[ex_bar]
    pnl_net = ret_gross - rt_cost
    y = (pnl_net > 0).astype(np.int64)
    if y.sum() < 50 or (1 - y).sum() < 50:
        return None

    feat = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    feat_names = list(feat.columns)
    X = feat.iloc[ev].to_numpy(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # label span in BAR space -> map to event space for purged CV label_span:
    # consecutive events are ~ (n / n_ev) bars apart; an event's label lives for
    # `hold` bars, so it overlaps ceil(hold / gap) later events. Use the median.
    gap = max(1, int(np.median(np.diff(ev)))) if len(ev) > 1 else 1
    label_span_ev = int(np.ceil(MAX_HOLD / gap)) + 1
    # event label bar-span (for uniqueness / sequential bootstrap), within events
    t0 = (np.asarray(ev, np.int64) - ev[0])
    t1 = np.minimum(t0 + hold.astype(np.int64), (n - 1) - ev[0])
    nb_span = int((n - 1) - ev[0]) + 1

    return dict(bars=bars, X=X, y=y, ev=ev, side=side, ret_gross=ret_gross,
                hold=hold, t0=t0, t1=t1, nb_span=nb_span, cost=rt_cost, bpy=bpy,
                feat_names=feat_names, label_span_ev=label_span_ev,
                n_bars=n, market=market)


# =========================================================================== #
# Sample weights (uniqueness-based, AFML Ch.4)
# =========================================================================== #
def make_sample_weights(t0, t1, nb_span):
    """Weight each event by its average label uniqueness (down-weights events
    whose label windows overlap many others). Normalised to mean 1."""
    u, _ = avg_uniqueness(t0, t1, nb_span)
    u = np.where(np.isfinite(u) & (u > 0), u, 1e-6)
    return u / u.mean(), float(u.mean())


# =========================================================================== #
# Models
# =========================================================================== #
def make_rf(params, avg_u, sample_weighted):
    """RandomForest-style bagged trees. LdP recipe (AFML 6.2/6.3):
      - low max_features (decorrelate trees)
      - min_weight_fraction_leaf > 0 (regularise leaves under weighting)
      - max_samples = average uniqueness (sequential-bootstrap analogue: each
        bootstrap draws ~ avg-uniqueness fraction so in-bag overlap drops).
    sklearn RF doesn't expose sequential bootstrap, but setting max_samples to
    the average uniqueness reproduces its first-order effect (fewer redundant
    in-bag draws); we ALSO build the true sequential-bootstrap bag via Bagging
    below for the headline model. Class balanced + sample weights supplied."""
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_features=params.get("max_features", 2),
        min_weight_fraction_leaf=params.get("min_weight_fraction_leaf", 0.0),
        max_samples=float(np.clip(avg_u, 0.05, 0.99)),
        bootstrap=True, oob_score=False, class_weight="balanced_subsample",
        n_jobs=1, random_state=RANDOM_STATE)


def make_hgb(params):
    return HistGradientBoostingClassifier(
        max_iter=params.get("max_iter", 200),
        learning_rate=params.get("learning_rate", 0.1),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        l2_regularization=1.0, early_stopping=False,
        random_state=RANDOM_STATE)


def _grid(grid: dict):
    import itertools
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


# =========================================================================== #
# Purged grid-search: score by neg-log-loss (headline) AND accuracy (control)
# =========================================================================== #
def purged_grid_search(make_model, grid, X, y, sw, label_span, score="nll",
                       n_splits=N_SPLITS, avg_u=None):
    """Return (best_params, best_score, all_scores). Score = mean over purged
    folds of (-log_loss) for 'nll' or accuracy for 'acc'. Trains with sample
    weights; evaluates on the purged OOS fold."""
    best, best_s, alls = None, -np.inf, []
    for params in _grid(grid):
        fold_s = []
        for tr, te in O.purged_kfold_splits(len(y), n_splits, EMBARGO, label_span):
            if len(tr) < 60 or y[tr].sum() < 5 or (1 - y[tr]).sum() < 5 \
               or len(np.unique(y[tr])) < 2:
                continue
            mdl = make_model(params, avg_u, True) if make_model is make_rf \
                else make_model(params)
            try:
                mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
            except TypeError:
                mdl.fit(X[tr], y[tr])
            if score == "nll":
                p = mdl.predict_proba(X[te])
                fold_s.append(-log_loss(y[te], p, labels=[0, 1]))
            else:
                fold_s.append(accuracy_score(y[te], mdl.predict(X[te])))
        if not fold_s:
            continue
        ms = float(np.mean(fold_s))
        alls.append((params, ms))
        if ms > best_s:
            best_s, best = ms, params
    return best, best_s, alls


# =========================================================================== #
# OOF probabilities + IS/OOS gap for a fixed model config
# =========================================================================== #
def oof_predict(make_model, params, X, y, sw, label_span, avg_u,
                n_splits=N_SPLITS):
    """Purged out-of-fold P(profit) and the IS-OOS log-loss gap (overfit gap)."""
    p_oof = np.full(len(y), np.nan)
    ll_is, ll_oos = [], []
    for tr, te in O.purged_kfold_splits(len(y), n_splits, EMBARGO, label_span):
        if len(tr) < 60 or len(np.unique(y[tr])) < 2:
            p_oof[te] = y[tr].mean() if len(tr) else 0.5
            continue
        mdl = make_model(params, avg_u, True) if make_model is make_rf \
            else make_model(params)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        pi = list(mdl.classes_).index(1) if 1 in mdl.classes_ else 0
        p_oof[te] = mdl.predict_proba(X[te])[: pi]
        ll_is.append(log_loss(y[tr], mdl.predict_proba(X[tr]), labels=[0, 1]))
        ll_oos.append(log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1]))
    p_oof = np.nan_to_num(p_oof, nan=float(y.mean()))
    gap = float(np.mean(ll_oos) - np.mean(ll_is)) if ll_is else np.nan
    return p_oof, gap, (float(np.mean(ll_is)) if ll_is else np.nan,
                        float(np.mean(ll_oos)) if ll_oos else np.nan)


# =========================================================================== #
# Bet P&L series + DSR
# =========================================================================== #
def trade_returns_to_series(ev, hold, pnl, n_bars):
    r = np.zeros(n_bars)
    for k in range(len(ev)):
        i0 = int(ev[k]); h = max(1, int(hold[k]))
        per = pnl[k] / h
        j1 = min(i0 + h, n_bars)
        r[i0:j1] += per
    return r


def score_bet(p_oof, ret_gross, ev, hold, cost, n_bars, bpy):
    act = (p_oof >= ACT_THRESH)
    size = np.where(act, p_oof, 0.0)
    # `cost` is the per-trade realistic ROUND-TRIP cost (entry+exit per-side).
    pnl = size * ret_gross - cost * size
    r = trade_returns_to_series(ev, hold, pnl, n_bars)
    rr = r[np.isfinite(r)]
    sr = O.sharpe(rr)
    sk = float(ss.skew(rr)) if len(rr) > 2 else 0.0
    ku = float(ss.kurtosis(rr, fisher=False)) if len(rr) > 2 else 3.0
    nz = rr[rr != 0]
    pf = float(nz[nz > 0].sum() / -nz[nz < 0].sum()) if (nz < 0).any() else np.nan
    return dict(r=r, sr=sr, sr_ann=sr * np.sqrt(bpy), skew=sk, kurt=ku,
                pf=pf, n_obs=len(rr), frac_act=float(act.mean()))


# =========================================================================== #
# Feature importance: MDI, MDA, clustered-MDA
# =========================================================================== #
def feature_clusters(X, feat_names, max_k=6):
    """Cluster features by |correlation| (single linkage on 1-|corr|).
    Returns list-of-lists of column indices."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    C = np.corrcoef(X.T)
    C = np.nan_to_num(C, nan=0.0)
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    np.fill_diagonal(D, 0.0)
    k = min(max_k, X.shape[1] - 1)
    try:
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=k, criterion="maxclust")
    except Exception:
        lab = np.arange(1, X.shape[1] + 1)
    clusters = [list(np.where(lab == c)[0]) for c in np.unique(lab)]
    return clusters


def importance_mdi(make_model, params, X, y, sw, avg_u):
    """MDI: mean impurity decrease from a single full-sample RF fit (in-sample,
    biased, exactly the LdP cautionary baseline)."""
    mdl = make_rf(params, avg_u, True)
    try:
        mdl.fit(X, y, sample_weight=sw)
    except TypeError:
        mdl.fit(X, y)
    imp = mdl.feature_importances_
    return imp / imp.sum() if imp.sum() > 0 else imp


def importance_mda(X, y, sw, label_span, avg_u, params, clustered=False,
                   clusters=None, n_splits=N_SPLITS, rng=None):
    """MDA (AFML 8.3 / clustered 8.5): for each purged fold, fit on train, score
    baseline OOS neg-log-loss, then permute each feature (or cluster of features)
    in the OOS block and measure the drop in score. Importance = mean normalised
    drop across folds. Scored by LOG-LOSS (probabilistic), not accuracy."""
    rng = rng or np.random.default_rng(RANDOM_STATE)
    p = X.shape[1]
    groups = clusters if clustered else [[j] for j in range(p)]
    drops = np.zeros((0, len(groups)))
    for tr, te in O.purged_kfold_splits(len(y), n_splits, EMBARGO, label_span):
        if len(tr) < 60 or len(np.unique(y[tr])) < 2 or len(te) < 10:
            continue
        mdl = make_rf(params, avg_u, True)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        base = -log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1])
        row = np.empty(len(groups))
        for gi, g in enumerate(groups):
            Xp = X[te].copy()
            perm = rng.permutation(len(te))
            for j in g:
                Xp[: j] = Xp[perm, j]
            sc = -log_loss(y[te], mdl.predict_proba(Xp), labels=[0, 1])
            row[gi] = (base - sc) / abs(base) if base != 0 else (base - sc)
        drops = np.vstack([drops, row])
    if drops.shape[0] == 0:
        return np.full(len(groups), np.nan), groups
    return drops.mean(0), groups


def importance_stability(X, y, sw, label_span, avg_u, params, clusters,
                         top_k=3):
    """Stability of the MDA top-set across CPCV paths (AFML Ch.12). For each CPCV
    path compute MDA, take the top-k features, and report the mean Jaccard overlap
    of those top-sets across paths (1 = perfectly stable selection)."""
    paths_top = []
    rng = np.random.default_rng(RANDOM_STATE)
    for tr, te in O.cpcv_splits(len(y), CPCV_GROUPS, CPCV_K, EMBARGO, label_span):
        if len(tr) < 80 or len(np.unique(y[tr])) < 2 or len(te) < 20:
            continue
        mdl = make_rf(params, avg_u, True)
        try:
            mdl.fit(X[tr], y[tr], sample_weight=sw[tr])
        except TypeError:
            mdl.fit(X[tr], y[tr])
        base = -log_loss(y[te], mdl.predict_proba(X[te]), labels=[0, 1])
        imp = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            Xp = X[te].copy()
            Xp[: j] = Xp[rng.permutation(len(te)), j]
            imp[j] = base - (-log_loss(y[te], mdl.predict_proba(Xp), labels=[0, 1]))
        paths_top.append(set(np.argsort(imp)[-top_k:]))
    if len(paths_top) < 2:
        return np.nan, len(paths_top)
    jac = []
    for i in range(len(paths_top)):
        for j in range(i + 1, len(paths_top)):
            a, b = paths_top[i], paths_top[j]
            jac.append(len(a & b) / len(a | b) if (a | b) else 0.0)
    return float(np.mean(jac)), len(paths_top)


# =========================================================================== #
# Core: run one instrument
# =========================================================================== #
def run_instrument(market, name, path, light=False, verbose=False):
    task = build_task(market, path)
    if task is None:
        return None
    X, y = task["X"], task["y"]
    sw, avg_u = make_sample_weights(task["t0"], task["t1"], task["nb_span"])
    ls = task["label_span_ev"]
    ev, hold, retg, cost, bpy = (task["ev"], task["hold"], task["ret_gross"],
                                 task["cost"], task["n_bars"])
    n_bars = task["n_bars"]; bpy = task["bpy"]
    fn = task["feat_names"]

    rf_grid = RF_GRID if not light else {"max_features": [2], "min_weight_fraction_leaf": [0.0], "n_estimators": [60]}
    hgb_grid = HGB_GRID if not light else {"max_iter": [60], "learning_rate": [0.1], "max_leaf_nodes": [15]}

    # ---- (1) BAGGING vs BOOSTING, tuned by NLL (headline) and ACC (control) ----
    rf_best_nll, rf_s_nll, rf_all = purged_grid_search(
        make_rf, rf_grid, X, y, sw, ls, "nll", avg_u=avg_u)
    hgb_best_nll, hgb_s_nll, hgb_all = purged_grid_search(
        make_hgb, hgb_grid, X, y, sw, ls, "nll", avg_u=avg_u)
    rf_best_acc, _, _ = purged_grid_search(
        make_rf, rf_grid, X, y, sw, ls, "acc", avg_u=avg_u)
    hgb_best_acc, _, _ = purged_grid_search(
        make_hgb, hgb_grid, X, y, sw, ls, "acc", avg_u=avg_u)

    # OOF predictions + overfit gap for the NLL-selected configs
    rf_p, rf_gap, rf_ll = oof_predict(make_rf, rf_best_nll, X, y, sw, ls, avg_u)
    hgb_p, hgb_gap, hgb_ll = oof_predict(make_hgb, hgb_best_nll, X, y, sw, ls, avg_u)
    # control: ACC-selected configs (for the IS-OOS gap comparison)
    rf_pa, rf_gapa, _ = oof_predict(make_rf, rf_best_acc, X, y, sw, ls, avg_u)
    hgb_pa, hgb_gapa, _ = oof_predict(make_hgb, hgb_best_acc, X, y, sw, ls, avg_u)

    rf_bet = score_bet(rf_p, retg, ev, hold, cost, n_bars, bpy)
    hgb_bet = score_bet(hgb_p, retg, ev, hold, cost, n_bars, bpy)

    # DSR: deflate each model's selected SR against the dispersion of the tuning-
    # grid bet SRs (the multiple-testing trials). Each grid config's OOF bet SR is
    # one trial; we pool the RF and HGB grids so both models deflate against the
    # same trial corpus (the full search the researcher ran).
    rf_trial_srs = _grid_bet_srs(make_rf, rf_grid, X, y, sw, ls, avg_u,
                                 retg, ev, hold, cost, n_bars, bpy)
    hgb_trial_srs = _grid_bet_srs(make_hgb, hgb_grid, X, y, sw, ls, avg_u,
                                  retg, ev, hold, cost, n_bars, bpy)
    all_trial_srs = np.array(rf_trial_srs + hgb_trial_srs)

    rf_dsr = O.deflated_sharpe_ratio(rf_bet["sr"], rf_bet["n_obs"], rf_bet["skew"],
                                     rf_bet["kurt"], all_trial_srs)
    hgb_dsr = O.deflated_sharpe_ratio(hgb_bet["sr"], hgb_bet["n_obs"], hgb_bet["skew"],
                                      hgb_bet["kurt"], all_trial_srs)

    # ---- (2) FEATURE IMPORTANCE: MDI vs MDA vs clustered-MDA ----
    clusters = feature_clusters(X, fn)
    mdi = importance_mdi(make_rf, rf_best_nll, X, y, sw, avg_u)
    mda, _ = importance_mda(X, y, sw, ls, avg_u, rf_best_nll, clustered=False)
    cmda, cgroups = importance_mda(X, y, sw, ls, avg_u, rf_best_nll,
                                   clustered=True, clusters=clusters)
    stab_mda, n_paths = importance_stability(X, y, sw, ls, avg_u, rf_best_nll,
                                             clusters, top_k=3)

    # MDI substitution-bias signal: corr between MDI rank and feature mutual
    # correlation magnitude (high-corr features hogging MDI). Spearman.
    Cmag = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)
    np.fill_diagonal(Cmag, 0.0)
    feat_corr = Cmag.mean(0)
    mdi_bias_rho = float(ss.spearmanr(mdi, feat_corr).correlation) if X.shape[1] > 3 else np.nan
    mda_bias_rho = float(ss.spearmanr(np.nan_to_num(mda), feat_corr).correlation) if X.shape[1] > 3 else np.nan

    return dict(
        market=market, name=name, n_bars=n_bars, n_ev=len(ev), bpy=bpy,
        base_rate=float(y.mean()), avg_uniqueness=avg_u,
        rf_best=rf_best_nll, hgb_best=hgb_best_nll,
        rf_best_acc=rf_best_acc, hgb_best_acc=hgb_best_acc,
        rf_sr_ann=rf_bet["sr_ann"], hgb_sr_ann=hgb_bet["sr_ann"],
        rf_pf=rf_bet["pf"], hgb_pf=hgb_bet["pf"],
        rf_dsr=rf_dsr["dsr"], hgb_dsr=hgb_dsr["dsr"], sr0=rf_dsr["sr0"],
        rf_frac_act=rf_bet["frac_act"], hgb_frac_act=hgb_bet["frac_act"],
        rf_gap_nll=rf_gap, hgb_gap_nll=hgb_gap,
        rf_gap_acc=rf_gapa, hgb_gap_acc=hgb_gapa,
        rf_ll_is=rf_ll[0], rf_ll_oos=rf_ll[1],
        hgb_ll_is=hgb_ll[0], hgb_ll_oos=hgb_ll[1],
        n_trials=len(all_trial_srs),
        mdi_bias_rho=mdi_bias_rho, mda_bias_rho=mda_bias_rho,
        stab_mda=stab_mda, n_cpcv_paths=n_paths,
        _feat_names=fn, _mdi=mdi, _mda=mda, _cmda=cmda, _cgroups=cgroups,
        _clusters=clusters, _rf_bet=rf_bet, _hgb_bet=hgb_bet)


def _grid_bet_srs(make_model, grid, X, y, sw, ls, avg_u, retg, ev, hold,
                  cost, n_bars, bpy):
    """OOF bet per-bar SR for every grid config (the DSR trial dispersion)."""
    out = []
    for params in _grid(grid):
        p, _, _ = oof_predict(make_model, params, X, y, sw, ls, avg_u)
        out.append(score_bet(p, retg, ev, hold, cost, n_bars, bpy)["sr"])
    return out


# =========================================================================== #
# Universe
# =========================================================================== #
def universe(smoke=False):
    u = []
    for p in sorted(glob.glob(os.path.join(CRYPTO_DIR, "*_1m.parquet"))):
        u.append(("crypto", os.path.basename(p).replace("_1m.parquet", ""), p))
    for s in ETF_SYMS:
        p = os.path.join(ETF_DIR, f"{s}.csv.gz")
        if os.path.exists(p):
            u.append(("equities", s, p))
    for p in sorted(glob.glob(os.path.join(FX_DIR, "*_fx1m.parquet"))):
        u.append(("forex", os.path.basename(p).replace("_fx1m.parquet", ""), p))
    if smoke:
        return [next(x for x in u if x[1] == "BTCUSDT"),
                next(x for x in u if x[1] == "SPY"),
                next(x for x in u if x[1] == "EURUSD")]
    return u


# =========================================================================== #
# Tables / figures
# =========================================================================== #
def make_tables(df):
    cols = ["market", "name", "n_ev", "base_rate", "avg_uniqueness",
            "rf_sr_ann", "hgb_sr_ann", "rf_pf", "hgb_pf",
            "rf_dsr", "hgb_dsr", "rf_gap_nll", "hgb_gap_nll",
            "rf_gap_acc", "hgb_gap_acc", "mdi_bias_rho", "mda_bias_rho",
            "stab_mda", "n_cpcv_paths"]
    t = df[[c for c in cols if c in df.columns]].round(4)
    t.to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)
    g = df.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_rf_dsr": g["rf_dsr"].median(),
        "med_hgb_dsr": g["hgb_dsr"].median(),
        "med_rf_gap_nll": g["rf_gap_nll"].median(),
        "med_hgb_gap_nll": g["hgb_gap_nll"].median(),
        "med_rf_gap_acc": g["rf_gap_acc"].median(),
        "med_hgb_gap_acc": g["hgb_gap_acc"].median(),
        "med_mdi_bias_rho": g["mdi_bias_rho"].median(),
        "med_mda_bias_rho": g["mda_bias_rho"].median(),
        "med_stab_mda": g["stab_mda"].median(),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("# Ensembles + Feature Importance, results\n\n## By-market\n")
        f.write(summ.to_markdown())
        f.write("\n\n## Per-instrument (head)\n")
        f.write(t.head(50).to_markdown(index=False))
    print("  tables ->", os.path.abspath(TAB))
    return summ


def make_figures(df, reps):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style(); P = style.PALETTE
    markets = ["crypto", "equities", "forex"]

    # FIG1: bagging vs boosting OOS DSR by market
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(markets)); w = 0.36
    rf = [df[df.market == m]["rf_dsr"].median() for m in markets]
    hg = [df[df.market == m]["hgb_dsr"].median() for m in markets]
    ax[0].bar(xs - w/2, rf, w, label="bagging (RF)", color=P["dollar"])
    ax[0].bar(xs + w/2, hg, w, label="boosting (HGB)", color=P["accent"])
    ax[0].axhline(0.95, color="k", ls="--", lw=0.8)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(markets); ax[0].set_ylabel("median OOS DSR")
    ax[0].set_title("Bagging vs boosting: DSR"); ax[0].legend()
    rfg = [df[df.market == m]["rf_gap_nll"].median() for m in markets]
    hgg = [df[df.market == m]["hgb_gap_nll"].median() for m in markets]
    ax[1].bar(xs - w/2, rfg, w, label="bagging (RF)", color=P["dollar"])
    ax[1].bar(xs + w/2, hgg, w, label="boosting (HGB)", color=P["accent"])
    ax[1].set_xticks(xs); ax[1].set_xticklabels(markets)
    ax[1].set_ylabel("IS-OOS log-loss gap"); ax[1].set_title("Overfit gap (lower=better)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_bagging_vs_boosting.png")); plt.close(fig)

    # FIG2: NLL vs ACC tuning overfit gap
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    gnll = df[["rf_gap_nll", "hgb_gap_nll"]].mean(1)
    gacc = df[["rf_gap_acc", "hgb_gap_acc"]].mean(1)
    ax.scatter(gacc, gnll, c=[{"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}[m] for m in df.market], s=22)
    lim = [min(gacc.min(), gnll.min()), max(gacc.max(), gnll.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("overfit gap, ACCURACY-tuned (control)")
    ax.set_ylabel("overfit gap, NEG-LOG-LOSS-tuned (headline)")
    ax.set_title("Tuning objective vs overfit gap")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_nll_vs_acc_tuning.png")); plt.close(fig)

    # FIG3: MDI vs MDA importance for a representative instrument per market
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    for i, m in enumerate(markets):
        if m not in reps:
            continue
        r = reps[m]; fn = r["_feat_names"]
        order = np.argsort(r["_mdi"])[::-1]
        ax[i].barh(np.arange(len(fn)) - 0.2, r["_mdi"][order], 0.4, color=P["accent"], label="MDI")
        mda_n = np.nan_to_num(r["_mda"]); mda_n = mda_n / (np.abs(mda_n).sum() or 1)
        ax[i].barh(np.arange(len(fn)) + 0.2, mda_n[order], 0.4, color=P["dollar"], label="MDA (norm)")
        ax[i].set_yticks(np.arange(len(fn))); ax[i].set_yticklabels([fn[j] for j in order], fontsize=8)
        ax[i].set_title(f"{m}: {r['name']}"); ax[i].axvline(0, color="k", lw=0.6)
        if i == 0: ax[i].legend()
    fig.suptitle("MDI (biased, in-sample) vs MDA (permutation, OOS)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_mdi_vs_mda.png")); plt.close(fig)

    # FIG4: MDA top-set stability across CPCV paths, by market
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
    for i, m in enumerate(markets):
        sub = df[df.market == m]["stab_mda"].dropna()
        ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.08, 0.08, len(sub)),
                   sub, s=20, color={"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}[m])
        if len(sub): ax.plot([i], [sub.median()], "_", ms=30, color="k")
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(markets)
    ax.set_ylabel("mean Jaccard of top-3 MDA set across CPCV paths")
    ax.set_title("Feature-selection stability (1 = perfectly stable)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_importance_stability.png")); plt.close(fig)
    print("  figures ->", os.path.abspath(FIG))


# =========================================================================== #
# Verification & profiling
# =========================================================================== #
def verify_kernels():
    print("=== numba == python parity (sequential bootstrap & avg uniqueness) ===")
    rng = np.random.default_rng(7)
    n_ev = 400
    starts = np.sort(rng.integers(0, 5000, n_ev)).astype(np.int64)
    spans = rng.integers(5, 80, n_ev).astype(np.int64)
    t0 = starts; t1 = np.minimum(starts + spans, 5099); nb = 5100
    u1, c1 = _avg_uniqueness_kernel(t0, t1, nb)
    u2, c2 = _avg_uniqueness_ref(t0, t1, nb)
    du = float(np.max(np.abs(u1 - u2))); dc = int(np.max(np.abs(c1 - c2)))
    print(f"  avg_uniqueness: max|Δu|={du:.3e}  max|Δcount|={dc}")
    ru = rng.random(300)
    b1 = _seqboot_kernel(t0, t1, nb, 300, ru)
    b2 = _seqboot_ref(t0, t1, nb, 300, ru)
    db = int(np.max(np.abs(b1 - b2)))
    print(f"  seq_bootstrap : max|Δindex|={db}  (identical={bool(db==0)})")
    assert du < 1e-12 and dc == 0 and db == 0, "PARITY FAILED"
    print("  PARITY OK (bit-identical)")


def profile_one():
    """cProfile one instrument with the FULL grids (realistic full-run sizing)."""
    import cProfile, pstats, io
    market, name, path = "crypto", "BTCUSDT", os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    verify_kernels()                       # also warms numba
    t = build_task(market, path)           # warm sklearn import paths cheaply
    print(f"  task: {len(t['ev'])} events, {t['n_bars']} bars, base_rate={t['y'].mean():.3f}")
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path, light=False, verbose=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
    print(s.getvalue())


# =========================================================================== #
# Main
# =========================================================================== #
def _worker(arg):
    market, name, path, light = arg
    try:
        r = run_instrument(market, name, path, light=light)
        return (market, name, r, None)
    except Exception as e:
        import traceback
        return (market, name, None, traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 3-instrument 1-core sanity")
    ap.add_argument("--profile", action="store_true", help="cProfile one instrument")
    ap.add_argument("--verify", action="store_true", help="numba parity checks only")
    ap.add_argument("--light", action="store_true", help="use tiny grids (smoke uses this)")
    ap.add_argument("--jobs", type=int, default=1, help="instrument-level process pool size")
    args = ap.parse_args()

    if args.verify:
        verify_kernels(); return
    if args.profile:
        profile_one(); return

    light = args.light or args.smoke
    u = universe(smoke=args.smoke)
    if args.smoke:
        verify_kernels()
    print(f"running {len(u)} instruments across {len(set(m for m,_,_ in u))} markets "
          f"(light={light}, jobs={args.jobs})")
    rows, reps = [], {}
    want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
    t0 = time.perf_counter()

    def _ingest(market, name, r, err, dt):
        if err is not None:
            print(f"  [ERR]  {market:9s} {name:10s}: {err.splitlines()[-1]}")
            if args.smoke: print(err)
            return
        if r is None:
            print(f"  [skip] {market:9s} {name:10s}"); return
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        if market not in reps or name == want.get(market):
            reps[market] = r
        print(f"  [ok]   {market:9s} {name:10s} "
              f"RF_DSR={r['rf_dsr']:.3f} HGB_DSR={r['hgb_dsr']:.3f} "
              f"RFgapNLL={r['rf_gap_nll']:.3f} HGBgapNLL={r['hgb_gap_nll']:.3f} "
              f"stab={r['stab_mda'] if r['stab_mda']==r['stab_mda'] else float('nan'):.2f} "
              f"({dt:.1f}s)")

    if args.jobs > 1 and not args.smoke:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for market, name, r, err in pool.imap_unordered(
                    _worker, [(m, n, p, light) for m, n, p in u]):
                _ingest(market, name, r, err, 0.0)
    else:
        for market, name, path in u:
            tt = time.perf_counter()
            _, _, r, err = _worker((market, name, path, light))
            _ingest(market, name, r, err, time.perf_counter() - tt)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results"); return
    df.to_parquet(os.path.join(TAB, "raw_results.parquet"))
    summ = make_tables(df)
    try:
        make_figures(df, reps)
    except Exception as e:
        print(f"  [fig ERR] {e}")
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===\n", summ.to_string())


if __name__ == "__main__":
    main()

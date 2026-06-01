#!/usr/bin/env python3
"""
run_uniqueness.py: Sample Uniqueness & Sequential Bootstrap at scale
(Lopez de Prado, AFML Ch.4), reproduced + extended across crypto, US equities,
and forex on real bars.

Pipeline per instrument:
  1. Load real 1m base -> ~N_TARGET dollar bars (crypto/equities) or tick bars (fx).
  2. Primary side = EMA(fast)/EMA(slow) crossover; events at crossover instants.
  3. Triple-barrier each event with full intrabar OHLC (reused Numba kernel from
     project 03). Each label spans [t0, t1] = [event bar, first-touch / vertical].
  4. Concurrency c_t, average uniqueness u_i, effective-N = sum u_i.   (Ch.4 core)
  5. Sequential bootstrap vs standard IID bootstrap: compare the average
     uniqueness of drawn samples (LdP's headline mechanical claim).
  6. Sample weights: uniqueness u_i and return-attribution w_i.
  7. DOES IT MATTER FOR A MODEL? Bagged trees, NAIVE (IID bag, no weights,
     max_samples=1.0) vs CORRECTED (sequential-bootstrap bags, max_samples ~=
     avg uniqueness, return-attribution sample weights). Compare OUT-OF-SAMPLE
     neg-log-loss / AUC, the IS->OOS gap, and feature-importance stability,
     under PURGED k-fold with embargo (no leakage).

Run:  python3 scripts/run_uniqueness.py            (full multi-market run)
      python3 scripts/run_uniqueness.py --smoke     (one instrument per market)
      python3 scripts/run_uniqueness.py --profile   (cProfile a single instrument)
"""
from __future__ import annotations
import sys, os, time, argparse, warnings, glob
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
# lib dir on path FIRST so the lib's numba-cached kernels can import 'bars'
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
# reuse the triple-barrier plumbing from the meta-labeling study
sys.path.insert(0, os.path.join(ROOT, "projects", "03_meta_labeling", "scripts"))

from lib import bars as B
from lib import overfit as O
import tbm
import uniqueness as U

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, roc_auc_score

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ----- data roots (same roots the meta-labeling and CV studies use) -----
CRYPTO_DIR = cfg.CRYPTO_1M
FX_DIR = cfg.FX_1M
ETF_DIR = cfg.EQUITY_1M
ETF_SYMS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

N_TARGET_BARS = 20000
VOL_SPAN = 50
N_SPLITS = 6
EMBARGO = 0.01
SEED = 0

# fixed structural config (this study is methodological; we hold the labeller
# fixed and vary the non-IID corrections, not the strategy knobs)
FAST, SLOW = 20, 60
PT, SL = 1.5, 1.0
MAX_HOLD = 50            # also the conservative label_span for purging

# Overlap-severity sweep. At this CUSUM event density the PT/SL barriers bind
# quickly, so the vertical (max_hold) barrier rarely sets the span; what governs
# span length (hence overlap) is the BARRIER WIDTH. We sweep a width multiplier
# applied to (PT, SL): wider barriers -> longer holds -> more overlap -> lower
# uniqueness. This is the informative "label-horizon" axis here.
WIDTH_SWEEP = [1.0, 2.0, 4.0]


# --------------------------------------------------------------------------- #
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


def label_spans(close, high, low, openp, vol, max_hold, width=1.0):
    """LdP-canonical labelling: CUSUM-filter event sampling (AFML Snippet 2.4)
    + triple-barrier with a max_hold vertical barrier. Returns event bar indices
    t0, label-end bar t1 (= first-touch / vertical), side, and the binary
    meta-label y = 1[gross side-return > 0].

    The CUSUM filter fires DENSELY (whenever |cumulative log-return| since the
    last event crosses a vol-scaled threshold), so spans of length up to
    max_hold overlap heavily: this is the non-IID-labels regime Ch.4 addresses.
    No look-ahead: the label of bar t0 resolves forward but is attributed to t0
    with its true [t0, t1] span; the side at t0 uses only closes <= t0.

    The CUSUM threshold is calibrated to the instrument's own return scale
    (median |bar return|), so event density is comparable across markets."""
    n = len(close)
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    med_abs = np.median(np.abs(log_ret[np.isfinite(log_ret) & (log_ret != 0)]))
    h = max(1e-9, 0.5 * med_abs)                     # ~ fires every few bars
    ev = U.cusum_events(log_ret, h)
    warm = SLOW + 12
    ev = ev[(ev > warm) & (ev < n - 1)]
    if len(ev) < 200:
        return None
    side_full = tbm.primary_ma_crossover(close, FAST, SLOW)
    side = side_full[ev]
    side[side == 0] = 1                              # default long if undefined
    tb = tbm.triple_barrier(close, high, low, openp, ev, side, vol,
                            PT * width, SL * width, max_hold)
    t0 = ev.astype(np.int64)
    t1 = tb["touch"].to_numpy(np.int64)              # first-touch / vertical bar
    t1 = np.maximum(t1, t0)                          # guard
    y = (tb["ret_gross"].to_numpy() > 0).astype(int)
    return dict(t0=t0, t1=t1, side=side, side_full=side_full, y=y,
                ret_gross=tb["ret_gross"].to_numpy())


# --------------------------------------------------------------------------- #
# bagged tree ensembles: naive (IID) vs corrected (seq-bootstrap + weights)
# --------------------------------------------------------------------------- #
def _fit_bag_naive(X, y, n_est, max_samples_frac, rng):
    """Standard IID bagging: each bag is a uniform with-replacement draw of
    floor(max_samples_frac * n) rows; unweighted trees."""
    n = len(y)
    m = max(10, int(round(max_samples_frac * n)))
    trees = []
    feat_imp = np.zeros(X.shape[1])
    for b in range(n_est):
        idx = rng.integers(0, n, size=m)
        if len(np.unique(y[idx])) < 2:
            continue
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                   random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx])
        trees.append(t)
        feat_imp += t.feature_importances_
    if not trees:
        return None
    return trees, feat_imp / len(trees)


def _fit_bag_corrected(X, y, t0, t1, n_bars, n_est, max_samples_frac,
                       sample_w, rng):
    """Corrected bagging: each bag drawn by SEQUENTIAL BOOTSTRAP (favours low
    overlap) of floor(max_samples_frac * n) rows; trees weighted by the
    return-attribution sample weights."""
    n = len(y)
    m = max(10, int(round(max_samples_frac * n)))
    trees = []
    feat_imp = np.zeros(X.shape[1])
    for b in range(n_est):
        idx = U.seq_bootstrap(t0, t1, n_bars, m, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                   random_state=int(rng.integers(0, 2**31)))
        t.fit(X[idx], y[idx], sample_weight=sample_w[idx])
        trees.append(t)
        feat_imp += t.feature_importances_
    if not trees:
        return None
    return trees, feat_imp / len(trees)


def _bag_proba(trees, X):
    p = np.zeros(len(X))
    for t in trees:
        cls = list(t.classes_)
        if len(cls) == 1:
            p += float(cls[0])
        else:
            p += t.predict_proba(X)[:, cls.index(1)]
    return p / len(trees)


def _safe_logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return log_loss(y, p, labels=[0, 1])


def _safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


# --------------------------------------------------------------------------- #
def run_instrument(market, name, path, n_est=40):
    bars = make_bars(market, path)
    if len(bars) < 3000:
        return None
    close = bars["close"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    openp = bars["open"].to_numpy(np.float64)
    n_bars = len(close)
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    vol = tbm.ewma_vol(close, VOL_SPAN)

    # ---- overlap-severity sweep over barrier width (Ch.4 core descriptive) ----
    horizon_rows = []
    for width in WIDTH_SWEEP:
        lab = label_spans(close, high, low, openp, vol, MAX_HOLD, width=width)
        if lab is None:
            continue
        t0, t1 = lab["t0"], lab["t1"]
        c = U.concurrency(t0, t1, n_bars)
        u = U.avg_uniqueness(t0, t1, c)
        N = len(t0)
        eff = float(u.sum())
        horizon_rows.append(dict(
            market=market, name=name, width=width, N=N,
            mean_avg_uniqueness=float(u.mean()),
            median_avg_uniqueness=float(np.median(u)),
            effective_N=eff, eff_ratio=eff / N,
            mean_concurrency=float(c[c > 0].mean()),
            max_concurrency=int(c.max()),
            mean_span=float((t1 - t0 + 1).mean())))

    # ---- the main config (MAX_HOLD) for bootstrap + model ----
    lab = label_spans(close, high, low, openp, vol, MAX_HOLD)
    if lab is None:
        return None
    t0, t1, y = lab["t0"], lab["t1"], lab["y"]
    side_full = lab["side_full"]
    N = len(t0)
    if y.sum() < 30 or (1 - y).sum() < 30:
        return None
    c = U.concurrency(t0, t1, n_bars)
    u = U.avg_uniqueness(t0, t1, c)
    eff = float(u.sum())
    w_ret = U.return_attribution_weights(t0, t1, c, log_ret)

    # ---- bootstrap uniqueness comparison ----
    rng = np.random.default_rng(SEED)
    n_draw = N
    seq_idx = U.seq_bootstrap(t0, t1, n_bars, n_draw, rng)
    std_idx = U.standard_bootstrap(N, n_draw, rng)
    seq_u = float(np.mean(u[seq_idx]))
    std_u = float(np.mean(u[std_idx]))

    # ---- model: naive vs corrected, purged k-fold OOS ----
    feat = tbm.build_features(bars, side_full, vol, FAST, SLOW)
    X = feat.iloc[t0].to_numpy(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feat_names = list(feat.columns)

    # average uniqueness sets the corrected bag size (LdP: avoid oversampling
    # redundant info -> max_samples ~ mean average uniqueness)
    ms_corr = float(np.clip(u.mean(), 0.05, 1.0))

    oos_ll_naive, oos_ll_corr = [], []
    oos_auc_naive, oos_auc_corr = [], []
    is_ll_naive, is_ll_corr = [], []
    fi_naive_folds, fi_corr_folds = [], []

    for tr, te in O.purged_kfold_splits(N, N_SPLITS, EMBARGO, label_span=3):
        if len(tr) < 80 or y[tr].sum() < 10 or (1 - y[tr]).sum() < 10:
            continue
        rng_n = np.random.default_rng(SEED + 1)
        rng_c = np.random.default_rng(SEED + 2)
        # local spans for the training subset (remap to a local bar axis so the
        # sequential bootstrap operates only over the train labels)
        lt0 = t0[tr]; lt1 = t1[tr]
        # local concurrency over the global bar axis is fine; seq bootstrap uses
        # the global n_bars. Build sample weights for the train rows.
        w_tr = w_ret[tr]
        w_tr = w_tr * (len(w_tr) / w_tr.sum()) if w_tr.sum() > 0 else np.ones(len(tr))

        rn = _fit_bag_naive(X[tr], y[tr], n_est, 1.0, rng_n)
        rc = _fit_bag_corrected(X[tr], y[tr], lt0, lt1, n_bars, n_est, ms_corr,
                                w_tr, rng_c)
        if rn is None or rc is None:
            continue
        trees_n, fi_n = rn
        trees_c, fi_c = rc

        pn_te = _bag_proba(trees_n, X[te])
        pc_te = _bag_proba(trees_c, X[te])
        pn_tr = _bag_proba(trees_n, X[tr])
        pc_tr = _bag_proba(trees_c, X[tr])

        oos_ll_naive.append(_safe_logloss(y[te], pn_te))
        oos_ll_corr.append(_safe_logloss(y[te], pc_te))
        oos_auc_naive.append(_safe_auc(y[te], pn_te))
        oos_auc_corr.append(_safe_auc(y[te], pc_te))
        is_ll_naive.append(_safe_logloss(y[tr], pn_tr))
        is_ll_corr.append(_safe_logloss(y[tr], pc_tr))
        fi_naive_folds.append(fi_n)
        fi_corr_folds.append(fi_c)

    if not oos_ll_naive:
        return None

    # feature-importance stability = mean pairwise rank correlation across folds
    def fi_stability(folds):
        F = np.array(folds)
        if len(F) < 2:
            return np.nan
        from scipy.stats import spearmanr
        rs = []
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                rho = spearmanr(F[i], F[j]).correlation
                if np.isfinite(rho):
                    rs.append(rho)
        return float(np.mean(rs)) if rs else np.nan

    oos_ll_n = float(np.mean(oos_ll_naive)); oos_ll_c = float(np.mean(oos_ll_corr))
    is_ll_n = float(np.mean(is_ll_naive)); is_ll_c = float(np.mean(is_ll_corr))

    return dict(
        market=market, name=name, n_bars=n_bars, N=N,
        mean_avg_uniqueness=float(u.mean()),
        median_avg_uniqueness=float(np.median(u)),
        effective_N=eff, eff_ratio=eff / N,
        mean_concurrency=float(c[c > 0].mean()), max_concurrency=int(c.max()),
        seq_boot_uniqueness=seq_u, std_boot_uniqueness=std_u,
        boot_uniqueness_lift=seq_u - std_u,
        ms_corrected=ms_corr,
        oos_logloss_naive=oos_ll_n, oos_logloss_corrected=oos_ll_c,
        oos_logloss_improve=oos_ll_n - oos_ll_c,
        oos_auc_naive=float(np.nanmean(oos_auc_naive)),
        oos_auc_corrected=float(np.nanmean(oos_auc_corr)),
        is_logloss_naive=is_ll_n, is_logloss_corrected=is_ll_c,
        gap_naive=oos_ll_n - is_ll_n, gap_corrected=oos_ll_c - is_ll_c,
        fi_stability_naive=fi_stability(fi_naive_folds),
        fi_stability_corrected=fi_stability(fi_corr_folds),
        # carry arrays for representative figures
        _u=u, _c=c, _seq_idx=seq_idx, _std_idx=std_idx,
        _fi_n=np.mean(fi_naive_folds, axis=0), _fi_c=np.mean(fi_corr_folds, axis=0),
        _feat_names=feat_names, _horizons=horizon_rows)


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
def profile_one():
    import cProfile, pstats, io
    market, name = "crypto", "BTCUSDT"
    path = os.path.join(CRYPTO_DIR, "BTCUSDT_1m.parquet")
    bars = make_bars(market, path)
    close = bars["close"].to_numpy(np.float64)
    vol = tbm.ewma_vol(close, VOL_SPAN)
    sf = tbm.primary_ma_crossover(close, FAST, SLOW); ev = tbm.crossover_events(sf)
    ev = ev[(ev > 72) & (ev < len(close) - 1)]
    # warm kernels
    _ = U.concurrency(ev[:3], ev[:3] + 5, len(close))
    _ = U.avg_uniqueness(ev[:3], ev[:3] + 5, U.concurrency(ev[:3], ev[:3] + 5, len(close)))
    _ = U.seq_bootstrap(ev[:5], ev[:5] + 5, len(close), 5, np.random.default_rng(0))
    pr = cProfile.Profile(); pr.enable()
    run_instrument(market, name, path)
    pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())


# --------------------------------------------------------------------------- #
def make_figures(df, reps, hdf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib import style
    style.set_style()
    P = style.PALETTE
    mkt_color = {"crypto": P["dollar"], "equities": P["tick"], "forex": P["volume"]}
    markets = [m for m in ["crypto", "equities", "forex"] if m in df.market.values]

    # FIG 1: average-uniqueness distribution per market (representative inst)
    fig, ax = plt.subplots(1, len(markets), figsize=(4.4 * len(markets), 3.9), squeeze=False)
    for i, m in enumerate(markets):
        u = reps[m]["_u"]
        ax[0][i].hist(u, bins=40, color=mkt_color[m], alpha=0.85)
        ax[0][i].axvline(u.mean(), color="k", ls="--", lw=1.1)
        ax[0][i].set_title(f"{m}: {reps[m]['name']}\nmean u = {u.mean():.3f}")
        ax[0][i].set_xlabel("average uniqueness u_i"); ax[0][i].set_ylabel("labels")
    fig.suptitle("Average-uniqueness distribution (overlap pushes u far below 1)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_avg_uniqueness_by_market.png"), dpi=130); plt.close(fig)

    # FIG 2: effective-N vs N per market (bar), with ratio annotation
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.4))
    g = df.groupby("market")
    xs = np.arange(len(markets)); w = 0.36
    Ns = [g.get_group(m)["N"].median() for m in markets]
    effs = [g.get_group(m)["effective_N"].median() for m in markets]
    ax.bar(xs - w/2, Ns, w, label="N (naive IID count)", color="#999999")
    ax.bar(xs + w/2, effs, w, label="effective N = sum u_i", color=P["accent"])
    for i, m in enumerate(markets):
        r = g.get_group(m)["eff_ratio"].median()
        ax.text(i, max(Ns[i], effs[i]) * 1.02, f"{r*100:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(markets)
    ax.set_ylabel("sample size (median per instrument)")
    ax.set_title("Effective sample size vs naive count (effective-N ratio annotated)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_effective_N_vs_N.png"), dpi=130); plt.close(fig)

    # FIG 3: sequential vs standard bootstrap uniqueness, per market
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.4))
    sb = [g.get_group(m)["std_boot_uniqueness"].median() for m in markets]
    qb = [g.get_group(m)["seq_boot_uniqueness"].median() for m in markets]
    ax.bar(xs - w/2, sb, w, label="standard IID bootstrap", color="#999999")
    ax.bar(xs + w/2, qb, w, label="sequential bootstrap", color=P["dollar"])
    ax.set_xticks(xs); ax.set_xticklabels(markets)
    ax.set_ylabel("avg uniqueness of drawn sample")
    ax.set_title("Sequential bootstrap raises sample uniqueness (LdP headline)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_seq_vs_std_bootstrap.png"), dpi=130); plt.close(fig)

    # FIG 4: IS->OOS gap, naive vs corrected, per market
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    gn = [g.get_group(m)["gap_naive"].median() for m in markets]
    gc = [g.get_group(m)["gap_corrected"].median() for m in markets]
    ax[0].bar(xs - w/2, gn, w, label="naive", color="#999999")
    ax[0].bar(xs + w/2, gc, w, label="corrected", color=P["accent"])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(markets)
    ax[0].set_ylabel("OOS - IS log-loss gap (lower = less overfit)")
    ax[0].set_title("IS->OOS overfit gap"); ax[0].legend()
    on = [g.get_group(m)["oos_logloss_naive"].median() for m in markets]
    oc = [g.get_group(m)["oos_logloss_corrected"].median() for m in markets]
    ax[1].bar(xs - w/2, on, w, label="naive", color="#999999")
    ax[1].bar(xs + w/2, oc, w, label="corrected", color=P["dollar"])
    ax[1].set_xticks(xs); ax[1].set_xticklabels(markets)
    ax[1].set_ylabel("OOS log-loss (lower = better)")
    ax[1].set_title("Out-of-sample generalization"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_isoos_gap_by_market.png"), dpi=130); plt.close(fig)

    # FIG 5: feature-importance stability, naive vs corrected
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.4))
    fn = [g.get_group(m)["fi_stability_naive"].median() for m in markets]
    fc = [g.get_group(m)["fi_stability_corrected"].median() for m in markets]
    ax.bar(xs - w/2, fn, w, label="naive", color="#999999")
    ax.bar(xs + w/2, fc, w, label="corrected", color=P["accent"])
    ax.set_xticks(xs); ax.set_xticklabels(markets)
    ax.set_ylabel("mean cross-fold rank corr of feature importances")
    ax.set_title("Feature-importance stability (higher = more reproducible)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_feature_importance_stability.png"), dpi=130); plt.close(fig)

    # FIG 6: overlap severity, uniqueness & effective-N ratio vs barrier width
    if hdf is not None and len(hdf):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
        for m in markets:
            g6 = hdf[hdf.market == m].groupby("width")
            w = g6["mean_avg_uniqueness"].median()
            sp = g6["mean_span"].median()
            ax[0].plot(w.index, w.values, "-o", color=mkt_color[m], label=m)
            ax[1].plot(sp.values, w.values, "-o", color=mkt_color[m], label=m)
        ax[0].set_xlabel("barrier width multiplier (wider = longer holds)")
        ax[0].set_ylabel("median mean avg uniqueness")
        ax[0].set_title("Wider barriers -> more overlap -> lower uniqueness")
        ax[0].legend()
        ax[1].set_xlabel("median label span (bars held)")
        ax[1].set_ylabel("median mean avg uniqueness")
        ax[1].set_title("Uniqueness falls as labels span more bars")
        ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_overlap_vs_barrier_width.png"), dpi=130); plt.close(fig)
    print("  figures written to", os.path.abspath(FIG))


# --------------------------------------------------------------------------- #
def make_tables(df, horizons_df):
    cols = ["market", "name", "N", "mean_avg_uniqueness", "median_avg_uniqueness",
            "effective_N", "eff_ratio", "mean_concurrency", "max_concurrency",
            "std_boot_uniqueness", "seq_boot_uniqueness", "boot_uniqueness_lift",
            "ms_corrected", "oos_logloss_naive", "oos_logloss_corrected",
            "oos_logloss_improve", "oos_auc_naive", "oos_auc_corrected",
            "gap_naive", "gap_corrected",
            "fi_stability_naive", "fi_stability_corrected"]
    t = df[cols].round(4)
    t.to_csv(os.path.join(TAB, "per_instrument.csv"), index=False)

    g = df.groupby("market")
    summ = pd.DataFrame({
        "n_inst": g.size(),
        "med_N": g["N"].median(),
        "med_mean_avg_uniqueness": g["mean_avg_uniqueness"].median(),
        "med_effective_N": g["effective_N"].median(),
        "med_eff_ratio": g["eff_ratio"].median(),
        "med_mean_concurrency": g["mean_concurrency"].median(),
        "med_std_boot_uniqueness": g["std_boot_uniqueness"].median(),
        "med_seq_boot_uniqueness": g["seq_boot_uniqueness"].median(),
        "med_boot_uniqueness_lift": g["boot_uniqueness_lift"].median(),
        "med_oos_logloss_naive": g["oos_logloss_naive"].median(),
        "med_oos_logloss_corrected": g["oos_logloss_corrected"].median(),
        "med_oos_logloss_improve": g["oos_logloss_improve"].median(),
        "med_oos_auc_naive": g["oos_auc_naive"].median(),
        "med_oos_auc_corrected": g["oos_auc_corrected"].median(),
        "med_gap_naive": g["gap_naive"].median(),
        "med_gap_corrected": g["gap_corrected"].median(),
        "med_fi_stability_naive": g["fi_stability_naive"].median(),
        "med_fi_stability_corrected": g["fi_stability_corrected"].median(),
        "n_oos_improved": g.apply(lambda x: int((x["oos_logloss_improve"] > 0).sum())),
        "n_gap_shrunk": g.apply(lambda x: int((x["gap_corrected"] < x["gap_naive"]).sum())),
    }).round(4)
    summ.to_csv(os.path.join(TAB, "by_market_summary.csv"))

    if len(horizons_df):
        hg = horizons_df.groupby(["market", "width"]).agg(
            n_inst=("name", "size"),
            med_mean_avg_uniqueness=("mean_avg_uniqueness", "median"),
            med_eff_ratio=("eff_ratio", "median"),
            med_mean_concurrency=("mean_concurrency", "median"),
            med_mean_span=("mean_span", "median")).round(4)
        hg.to_csv(os.path.join(TAB, "uniqueness_by_horizon.csv"))

    md = ["# Sample Uniqueness & Sequential Bootstrap: results\n",
          "\n## By-market summary\n", summ.to_markdown(),
          "\n\n## Per-instrument\n", t.to_markdown(index=False)]
    with open(os.path.join(TAB, "results.md"), "w") as f:
        f.write("\n".join(md))
    print("  tables written to", os.path.abspath(TAB))
    return summ


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    if args.profile:
        profile_one(); return

    u = universe(smoke=args.smoke)
    print(f"running {len(u)} instruments across {len(set(m for m,_,_ in u))} markets")
    rows, horizon_rows, reps = [], [], {}
    t0 = time.perf_counter()
    # Per-instrument work is independent and each instrument seeds its own RNG
    # from the fixed SEED, so running them in parallel is bit-identical to serial.
    # Capped worker count keeps peak RAM well under the box limit.
    from joblib import Parallel, delayed

    def _one(market, name, path):
        try:
            tt = time.perf_counter()
            r = run_instrument(market, name, path)
            return (market, name, r, time.perf_counter() - tt, None)
        except Exception as e:
            import traceback
            return (market, name, None, 0.0, traceback.format_exc())

    n_jobs = min(16, len(u), max(1, (os.cpu_count() or 4) - 2))
    print(f"  running in parallel on {n_jobs} workers")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(market, name, path) for market, name, path in u
    )

    want = {"crypto": "BTCUSDT", "equities": "SPY", "forex": "EURUSD"}
    # Iterate in the original universe order so `reps` selection is deterministic.
    for market, name, r, dt, err in results:
        if err is not None:
            print(f"  [ERR]  {market:9s} {name:10s}: {err.splitlines()[-1]}"); continue
        if r is None:
            print(f"  [skip] {market:9s} {name:10s}"); continue
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        horizon_rows.extend(r["_horizons"])
        if market not in reps or name == want.get(market):
            reps[market] = r
        print(f"  [ok]   {market:9s} {name:10s} "
              f"u={r['mean_avg_uniqueness']:.3f} effN/N={r['eff_ratio']:.3f} "
              f"seqU={r['seq_boot_uniqueness']:.3f} stdU={r['std_boot_uniqueness']:.3f} "
              f"OOSll n={r['oos_logloss_naive']:.4f} c={r['oos_logloss_corrected']:.4f} "
              f"({dt:.1f}s)")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results"); return
    df.to_parquet(os.path.join(TAB, "raw_results.parquet"))
    hdf = pd.DataFrame(horizon_rows)
    hdf.to_parquet(os.path.join(TAB, "horizon_results.parquet"))
    summ = make_tables(df, hdf)
    make_figures(df, reps, hdf)
    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s")
    print("\n=== BY-MARKET SUMMARY ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()

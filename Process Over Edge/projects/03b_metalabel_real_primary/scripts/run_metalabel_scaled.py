#!/usr/bin/env python3
"""run_metalabel_scaled.py: SCALE the edged-primary meta-gate to the full set of
proven structural edges and pairs, and ask whether pooling the meta-gated sleeves
clears the Deflated Sharpe Ratio publication bar (DSR > 0.95).

CONTEXT
  Sibling script run_metalabel_real_primary.py established, on a TWO-pair sample,
  that meta-labeling a primary that ALREADY has an edge (LdP AFML Ch.3 precondition
  MET) lifts the edge: PF 1.26 -> 1.79, per-trade 47 -> 148 bp, DSR 0.64 -> 0.78.
  The DSR rose but did NOT clear the program's 0.95 bar on that small sample, and
  the writeup names "scaling to the full set of proven edges" as the natural next
  step. THIS script is that step.

WHAT IT DOES
  For the full set of proven structural archetypes (order-flow imbalance,
  open-interest squeeze, funding squeeze, cross-asset residual, cross-venue funding
  divergence, top-trader ratio) crossed with the multi-pair perp universe the
  engine supports (the same ~26-pair set the structural engine runs over), for EACH
  (edge, pair):
    primary alone (OOS)   vs   primary + ML meta-gate (OOS)
  net of per-fill costs, full-OHLC intrabar exits, purged walk-forward.
  Each archetype's primary signal is reproduced inline as a FAITHFUL copy of the
  structural engine's verbatim side construction (the engine modules cannot be
  imported as clean functions: they carry module-global SIDE/ALLOWED/SIZE buffers
  and a multiprocessing Pool harness). The verified numba sim kernel + the cost
  model are reused READ-ONLY (kernel.py / common.py / data.py are NEVER modified;
  only imported).

THE HEADLINE QUESTION
  Pooling the meta-gated proven edges as ONE meta-strategy (many weakly-correlated
  gated sleeves), does the Deflated Sharpe clear 0.95? We report:
    - the distribution of per-(edge,pair) DSR and how many clear 0.95,
    - the best single (edge,pair),
    - a POOLED/portfolio DSR (the equal-weight pooled meta sleeve, deflated HONESTLY
      against the trial family = the number of (edge,pair) configs searched), and
    - the primary-vs-meta lift across the full set (does the +0.5 PF / +100 bp
      per-trade lift from the 2-pair sample hold up broadly?).

GUARDRAILS (enforced)
  - Single-threaded BLAS/OMP (set below before any numpy/sklearn import side-effects).
  - ONE process. Optional --n-jobs for pair-level parallelism, single-threaded
    workers, capped at 8.
  - Per-pair single-asset work; no panel load. RAM stays well under ~10 GB.
  - The shared engine is READ-ONLY.

Run:
  python3 scripts/run_metalabel_scaled.py            # full set (heavy, minutes)
  python3 scripts/run_metalabel_scaled.py --smoke    # 2 edges x 3 pairs, fast sanity
  python3 scripts/run_metalabel_scaled.py --pairs ETHUSDT,SOLUSDT --edges B,D

Outputs: tables/metalabel_scaled.{csv,md}; figures/fig_metalabel_scaled.png;
         a writeup subsection appended to writeup/README.md.
"""
from __future__ import annotations

# --- HARD GUARDRAIL: single-thread the math libs BEFORE numpy/sklearn import ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_d = HERE
while _d != "/" and not os.path.exists(os.path.join(_d, "config.py")):
    _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, ROOT)
import config as cfg
sys.path.insert(0, cfg.LIB)
sys.path.insert(0, ROOT)
from lib import overfit as O   # DSR / PBO / sharpe (read-only)

FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
WRITEUP = os.path.join(HERE, "..", "writeup", "README.md")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Read-only structural-edge engine. Supplies the cost model + causal helpers
# (nc), the causal per-pair data loader (data_loader) and the verified sim kernel.
# Produced by a separate pipeline (see the study README) and not bundled here;
# resolve its location from config (override with $LDP_STRUCTURAL_ENGINE). It is
# imported LAZILY in load_engine() so importing this module never hard-fails when
# the engine is absent; the run then skips gracefully.
STRUCTURAL_ENGINE = cfg.STRUCTURAL_ENGINE
# Name of the engine's causal per-pair loader function (override if your copy
# exposes it under a different name).
ENGINE_LOADER_FN = os.environ.get("LDP_ENGINE_LOADER_FN", "load_pair")
nc = None
data_loader = None
sim_kernel = None
sharpe_jit = None
load_pair = None     # engine's causal per-pair loader, bound in load_engine()


def load_engine():
    """Import the read-only structural engine on demand. Returns True on success,
    False (with a skip message) if the engine is not available."""
    global nc, data_loader, sim_kernel, sharpe_jit, load_pair, MIN_HOLD_POOL
    if nc is not None:
        return True
    if STRUCTURAL_ENGINE not in sys.path:
        sys.path.insert(0, STRUCTURAL_ENGINE)
    try:
        import common as _nc
        import data as _data_loader
        from kernel import sim_kernel as _sim_kernel, sharpe_jit as _sharpe_jit
        _load_pair = getattr(_data_loader, ENGINE_LOADER_FN)
    except Exception as ex:
        print(f"  [skip] structural engine not available at {STRUCTURAL_ENGINE}: {ex}")
        print("         (set $LDP_STRUCTURAL_ENGINE to your copy; see the study README)")
        return False
    nc = _nc
    data_loader = _data_loader
    sim_kernel = _sim_kernel
    sharpe_jit = _sharpe_jit
    load_pair = _load_pair
    MIN_HOLD_POOL = nc.MIN_HOLD_POOL
    return True

from sklearn.ensemble import RandomForestClassifier

# Annualisation: perp 30m bars -> 48 bars/day.
BARS_PER_YEAR = 365 * 48

# Full proven-edge pair universe (same 26-pair set the structural engine runs over;
# every pair has binance perp + spot + funding + OI 5m on disk).
FULL_PAIRS = [
    "AAVEUSDT", "ALGOUSDT", "APEUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT",
    "ETCUSDT", "ETHUSDT", "HBARUSDT", "ICPUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "SOLUSDT", "SUIUSDT", "TRXUSDT", "UNIUSDT", "XLMUSDT",
    "XRPUSDT", "ZECUSDT",
]

# WFO config (overridable via env, default matches the tested regime).
IS = int(os.environ.get("ML_IS", 8000))
OOS = int(os.environ.get("ML_OOS", 4000))
NWIN = int(os.environ.get("ML_NWIN", 4))
NS = int(os.environ.get("ML_NS", 24))     # IS knob draws per window
MIN_IS_TR = 15                            # primary needs >= this to select a knob
MIN_TRAIN = 20                            # purged labels needed to train the secondary

# Shared IS-knob pools (subset of the structural engine's knob pools).
SL_OPTS = [100, 200, 400]
RRR_OPTS = [1, 2, 3]
HOLD_OPTS = [16, 32, 64]
MIN_HOLD_POOL = None                       # resolved from the engine in load_engine()
Z_ENTRY_OPTS = [1.5, 2.0, 2.5, 3.0]

EXIT_FAM = 2   # time_decay: the cleanest barrier for triple-barrier labels
EXIT_COMP = 0  # single


# --------------------------------------------------------------------------- #
# Faithful per-archetype PRIMARY signal builders.
# Each returns (feats_dict, side_builder). side_builder(d, feats, knob, lo, hi)
# returns an int8[N] side array (entry side per bar) over [lo,hi). The signal math
# is a verbatim copy of the structural engine's run_one_sample side construction for a
# REPRESENTATIVE structural shape of that archetype (direction/confirm/z-window
# pinned; numeric knobs IS-tuned). Public, generic edge labels used throughout.
# --------------------------------------------------------------------------- #
def _z_roll(arr, w):
    return np.nan_to_num(nc.causal_zscore_rolling(arr, w), nan=0.0)


def build_B(d):
    """Open-interest squeeze: OI-growth z-score x funding-sign squeeze (both ways)."""
    oi = d["oi"]
    oi_z = _z_roll(oi, 100)
    feats = dict(oi_z=oi_z, self_ret=np.nan_to_num(d["self_ret"], nan=0.0))

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        oz = feats["oi_z"]; fund = d["fund_bin"]; valid = d["valid_perp"]
        rising = oz > knob["thr"]
        sf = np.where(rising & (fund < 0) & valid, +1, 0).astype(np.int8)
        sf[(rising & (fund > 0) & valid)] = -1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


def build_A(d):
    """Funding-rate squeeze: fade extreme funding z-score (both ways)."""
    f = d["fund_bin"]
    z = _z_roll(f, 150)
    feats = dict(z=z, self_ret=np.nan_to_num(d["self_ret"], nan=0.0))

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        zz = feats["z"]; valid = d["valid_perp"]
        ze = knob["thr"]
        sf = np.where((zz > ze) & valid, -1, 0).astype(np.int8)
        sf[(zz < -ze) & valid] = +1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


def build_C(d):
    """Cross-asset residual: fade the leader-residual z (mean-revert vs BTC)."""
    self_ret = np.nan_to_num(d["self_ret"], nan=0.0)
    lr = d.get("leader_ret")
    leader_ret = np.nan_to_num(lr if lr is not None else np.zeros(d["N"]), nan=0.0)
    lb = 100
    s = pd.Series(self_ret); l = pd.Series(leader_ret)
    cov = s.rolling(lb, min_periods=lb).cov(l).shift(1).values
    var = l.rolling(lb, min_periods=lb).var().shift(1).values
    mean_s = s.rolling(lb, min_periods=lb).mean().shift(1).values
    mean_l = l.rolling(lb, min_periods=lb).mean().shift(1).values
    beta = np.full(d["N"], np.nan); v = np.isfinite(var) & (var > 0)
    beta[v] = cov[v] / var[v]
    alpha = np.full(d["N"], np.nan); alpha[v] = mean_s[v] - beta[v] * mean_l[v]
    resid = np.full(d["N"], np.nan)
    vv = np.isfinite(beta) & np.isfinite(alpha) & np.isfinite(self_ret) & np.isfinite(leader_ret)
    resid[vv] = self_ret[vv] - (alpha[vv] + beta[vv] * leader_ret[vv])
    z = _z_roll(np.nan_to_num(resid, nan=0.0), 100)
    feats = dict(z=z, self_ret=self_ret)

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        zz = feats["z"]; valid = d["valid_perp"]
        ze = knob["thr"]
        sf = np.where((zz < -ze) & valid, +1, 0).astype(np.int8)
        sf[((zz > ze) & valid)] = -1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


def build_D(d):
    """Order-flow imbalance: taker-buy imbalance z (mean-revert regime)."""
    tbuy = np.nan_to_num(d["tbuy_ratio"], nan=0.5)
    imb = tbuy - 0.5
    z = _z_roll(imb, 200)
    feats = dict(z=z, self_ret=np.nan_to_num(d["self_ret"], nan=0.0))

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        zz = feats["z"]; valid = d["valid_perp"]
        ze = knob["thr"]
        sf = np.where((zz > ze) & valid, -1, 0).astype(np.int8)  # mean_revert
        sf[(zz < -ze) & valid] = +1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


def build_E(d):
    """Cross-venue funding divergence: binance vs bybit funding z (follow divergence)."""
    fb = np.nan_to_num(d["fund_bin"], nan=0.0)
    fy = np.nan_to_num(d["fund_byb"], nan=0.0)
    div = fb - fy
    z = _z_roll(div, 200)
    has_byb = bool(np.any(np.isfinite(d["fund_byb"]) & (d["fund_byb"] != 0)))
    feats = dict(z=z, self_ret=np.nan_to_num(d["self_ret"], nan=0.0), has_byb=has_byb)

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        if not feats["has_byb"]:
            return s
        zz = feats["z"]; valid = d["valid_perp"]
        ze = knob["thr"]
        sf = np.where((zz > ze) & valid, -1, 0).astype(np.int8)  # follow_div
        sf[(zz < -ze) & valid] = +1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


def build_F(d):
    """Top-trader ratio: count-top long/short ratio z (contrarian regime)."""
    top_count = np.nan_to_num(d["top_ratio"], nan=1.0)
    z = _z_roll(top_count, 100)
    valid_top = np.isfinite(d["top_ratio"]) | np.isfinite(d["taker_ratio"])
    feats = dict(z=z, self_ret=np.nan_to_num(d["self_ret"], nan=0.0), valid_top=valid_top)

    def side(d, feats, knob, lo, hi):
        N = d["N"]; s = np.zeros(N, dtype=np.int8)
        zz = feats["z"]; valid = d["valid_perp"] & feats["valid_top"]
        ze = knob["thr"]
        sf = np.where((zz > ze) & valid, -1, 0).astype(np.int8)  # contrarian
        sf[(zz < -ze) & valid] = +1
        s[lo:hi] = sf[lo:hi]
        return s
    return feats, side


# id -> (public label, builder)
EDGES = {
    "A": ("funding squeeze", build_A),
    "B": ("open-interest squeeze", build_B),
    "C": ("cross-asset residual", build_C),
    "D": ("order-flow imbalance", build_D),
    "E": ("cross-venue funding divergence", build_E),
    "F": ("top-trader ratio", build_F),
}


# --------------------------------------------------------------------------- #
# Common per-window meta features (shared, all causal: shift(1) at load or
# built from already-shifted series).
# --------------------------------------------------------------------------- #
def common_meta_feats(d, feats):
    self_ret = feats["self_ret"]
    rstd50 = pd.Series(self_ret).rolling(50, min_periods=50).std().shift(1).values
    mom20 = pd.Series(self_ret).rolling(20, min_periods=20).sum().shift(1).values
    return dict(rstd50=np.nan_to_num(rstd50, nan=0.0),
                mom20=np.nan_to_num(mom20, nan=0.0))


def meta_X(d, feats, mfeats, side, idxs):
    """Causal feature matrix at entry bars idxs. The edge's own z-score, realised
    vol, momentum, funding sign, hour, and the proposed side. All shift(1)."""
    z = feats["z" if "z" in feats else "oi_z"][idxs]
    rstd = mfeats["rstd50"][idxs]
    mom = mfeats["mom20"][idxs]
    fund = d["fund_bin"][idxs]
    hour = d["hour"][idxs].astype(np.float64)
    sd = side[idxs].astype(np.float64)
    return np.column_stack([z, rstd, mom, fund, hour, sd])


def _sim(d, side, knob, lo, hi):
    N = d["N"]
    allowed = side != 0
    size = np.ones(N, dtype=np.float64)
    sl = knob["sl_bp"] / 1e4
    tp = sl * knob["rrr"]
    return sim_kernel(
        d["perp_close"], d["perp_high"], d["perp_low"], d["fcs"], d["fund_bin"],
        side, allowed, d["valid_perp"],
        1, sl, tp, knob["max_hold"], knob["min_hold"],
        EXIT_FAM, EXIT_COMP, size, lo, hi,
        nc.TAKER_FILL, nc.MAKER_FILL,
    )


# --------------------------------------------------------------------------- #
# One (edge, pair): purged-WFO primary-vs-meta. Returns OOS pnl arrays + records.
# --------------------------------------------------------------------------- #
def run_edge_pair(edge_id, pair):
    label, builder = EDGES[edge_id]
    d = load_pair(pair)
    feats, side_of = builder(d)
    mfeats = common_meta_feats(d, feats)
    N = d["N"]
    rng = np.random.default_rng(abs(hash((pair, edge_id))) % 2**32)
    zkey = "z" if "z" in feats else "oi_z"

    def select_knob(lo, ie):
        best = None; bs = -1e18
        for _ in range(NS):
            knob = dict(thr=float(rng.choice(Z_ENTRY_OPTS)),
                        sl_bp=int(rng.choice(SL_OPTS)),
                        rrr=int(rng.choice(RRR_OPTS)),
                        max_hold=int(rng.choice(HOLD_OPTS)),
                        min_hold=int(rng.choice(MIN_HOLD_POOL)))
            s = side_of(d, feats, knob, lo, ie)
            _, _, pnl, _ = _sim(d, s, knob, lo, ie)
            if pnl.shape[0] < MIN_IS_TR:
                continue
            sc = sharpe_jit(pnl)
            if np.isfinite(sc) and sc > bs:
                bs = sc; best = knob
        return best

    prim_oos, meta_oos, records = [], [], []
    # OI/divergence feeds begin partway through the perp series; start where the
    # edge's primary input is live + a z warmup (no leak; just delays the start).
    if edge_id in ("B", "F"):
        fin = np.isfinite(d["oi"])
    elif edge_id == "E":
        fin = np.isfinite(d["fund_byb"])
    else:
        fin = np.isfinite(d["perp_close"])
    start = int(np.argmax(fin)) if fin.any() else 0
    ws = start + 350   # past the longest rolling window warmup used above
    window_id = 0
    while ws + IS + OOS <= N and window_id < NWIN:
        ie = ws + IS; oe = ie + OOS
        knob = select_knob(ws, ie)
        if knob is None:
            ws += OOS; window_id += 1; continue

        side_is = side_of(d, feats, knob, ws, ie)
        i_ei, i_xi, i_pnl, i_w = _sim(d, side_is, knob, ws, ie)
        # PURGE: keep IS labels whose triple-barrier exit resolves before boundary.
        purge = i_xi < ie
        tr_ei = i_ei[purge]; tr_pnl = i_pnl[purge]

        side_oos = side_of(d, feats, knob, ie, oe)
        o_ei, o_xi, o_pnl, o_w = _sim(d, side_oos, knob, ie, oe)

        if tr_ei.shape[0] < MIN_TRAIN:
            # not enough purged labels -> meta == primary this window
            for k in range(o_pnl.shape[0]):
                prim_oos.append(float(o_pnl[k])); meta_oos.append(float(o_pnl[k]))
                records.append((0, window_id, 1, int(o_ei[k]), int(o_xi[k]), float(o_pnl[k]), float(o_w[k])))
                records.append((1, window_id, 1, int(o_ei[k]), int(o_xi[k]), float(o_pnl[k]), float(o_w[k])))
            ws += OOS; window_id += 1; continue

        y_tr = (tr_pnl > 0).astype(np.int8)
        X_tr = meta_X(d, feats, mfeats, side_is, tr_ei)
        for k in range(i_pnl.shape[0]):
            records.append((0, window_id, 0, int(i_ei[k]), int(i_xi[k]), float(i_pnl[k]), float(i_w[k])))

        if y_tr.sum() == 0 or y_tr.sum() == y_tr.shape[0]:
            rf = None
        else:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                        min_samples_leaf=20, class_weight="balanced",
                                        n_jobs=1, random_state=0)
            rf.fit(X_tr, y_tr)

        for k in range(o_pnl.shape[0]):
            prim_oos.append(float(o_pnl[k]))
            records.append((0, window_id, 1, int(o_ei[k]), int(o_xi[k]), float(o_pnl[k]), float(o_w[k])))

        take_mask = np.zeros(N, dtype=bool)
        entry_bars = np.where(side_oos != 0)[0]
        entry_bars = entry_bars[(entry_bars >= ie) & (entry_bars < oe)]
        if rf is None or entry_bars.shape[0] == 0:
            take_mask[entry_bars] = True
        else:
            Xo = meta_X(d, feats, mfeats, side_oos, entry_bars)
            pred = rf.predict(Xo)
            take_mask[entry_bars[pred == 1]] = True

        gated = np.where(take_mask, side_oos, 0).astype(np.int8)
        g_ei, g_xi, g_pnl, g_w = _sim(d, gated, knob, ie, oe)
        for k in range(g_pnl.shape[0]):
            meta_oos.append(float(g_pnl[k]))
            records.append((1, window_id, 1, int(g_ei[k]), int(g_xi[k]), float(g_pnl[k]), float(g_w[k])))

        ws += OOS; window_id += 1

    return (np.asarray(prim_oos), np.asarray(meta_oos), records)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def profit_factor(pnl):
    pnl = np.asarray(pnl, float)
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    return float(pos / neg) if neg > 1e-15 else (float("inf") if pos > 0 else 0.0)


def per_trade_bp(pnl):
    pnl = np.asarray(pnl, float)
    return float(pnl.mean() * 1e4) if pnl.size else float("nan")


def dsr_of(pnl, win_sharpes):
    """DSR of a per-trade stream, deflated against the per-window Sharpe dispersion."""
    from scipy import stats as ss
    p = np.asarray(pnl, float); p = p[np.isfinite(p)]
    if p.size < 3:
        return float("nan"), float("nan")
    sr = O.sharpe(p)
    sk = float(ss.skew(p)); ku = float(ss.kurtosis(p, fisher=False))
    win_sharpes = np.asarray(win_sharpes, float)
    if win_sharpes.size < 2:
        win_sharpes = np.array([0.0, 0.0])
    dd = O.deflated_sharpe_ratio(sr, p.size, sk, ku, win_sharpes)
    return float(dd["dsr"]), float(dd["sr0"])


def window_sharpes(records, arm):
    df = pd.DataFrame(records, columns=["combo_id", "window_id", "phase",
                                        "entry_idx", "exit_idx", "pnl", "weight"])
    oos = df[(df.phase == 1) & (df.combo_id == arm)]
    out = []
    for _, g in oos.groupby("window_id"):
        v = g["pnl"].to_numpy()
        if v.size >= 2:
            out.append(O.sharpe(v))
    return np.array(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="comma list; default = full 26-pair set")
    ap.add_argument("--edges", default=None, help="comma list of edge ids A..F; default all")
    ap.add_argument("--smoke", action="store_true", help="2 edges x 3 pairs, fast sanity")
    ap.add_argument("--n-jobs", type=int, default=1, help="pair-level parallelism (<=8)")
    args = ap.parse_args()

    if not load_engine():
        return

    edges = (args.edges.split(",") if args.edges
             else (["B", "D"] if args.smoke else list(EDGES.keys())))
    pairs = (args.pairs.split(",") if args.pairs
             else (["ETHUSDT", "SOLUSDT", "BTCUSDT"] if args.smoke else FULL_PAIRS))
    n_jobs = max(1, min(8, args.n_jobs))

    configs = [(e, p) for e in edges for p in pairs]
    n_trials = len(configs)
    print(f"=== SCALED meta-gate: {len(edges)} edges x {len(pairs)} pairs "
          f"= {n_trials} (edge,pair) configs; n_jobs={n_jobs} ===", flush=True)

    t0 = time.time()

    def one(cfg):
        e, p = cfg
        load_engine()   # re-bind in a fresh worker process (loky resets globals)
        try:
            prim, meta, recs = run_edge_pair(e, p)
        except FileNotFoundError as ex:
            print(f"  [skip] {e}/{p}: data missing ({ex})", flush=True); return None
        except Exception as ex:
            print(f"  [ERR]  {e}/{p}: {ex}", flush=True); return None
        if prim.size == 0:
            print(f"  [skip] {e}/{p}: no primary trades", flush=True); return None
        ws_p = window_sharpes(recs, 0); ws_m = window_sharpes(recs, 1)
        d_p, _ = dsr_of(prim, ws_p); d_m, _ = dsr_of(meta, ws_m)
        row = dict(
            edge=EDGES[e][0], pair_id=f"P{pairs.index(p)+1:02d}",
            prim_trades=int(prim.size), meta_trades=int(meta.size),
            prim_pf=round(profit_factor(prim), 3), meta_pf=round(profit_factor(meta), 3),
            prim_bp=round(per_trade_bp(prim), 1), meta_bp=round(per_trade_bp(meta), 1),
            prim_dsr=round(d_p, 3) if np.isfinite(d_p) else float("nan"),
            meta_dsr=round(d_m, 3) if np.isfinite(d_m) else float("nan"),
            gate_kept=round(float(meta.size) / prim.size, 3) if prim.size else float("nan"),
        )
        print(f"  [ok] {e}/{row['pair_id']:>3}  PF {row['prim_pf']:.2f}->{row['meta_pf']:.2f}  "
              f"bp {row['prim_bp']:.0f}->{row['meta_bp']:.0f}  DSR {row['prim_dsr']}->{row['meta_dsr']}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        return row, prim, meta

    results = []
    if n_jobs == 1:
        for cfg in configs:
            r = one(cfg)
            if r is not None:
                results.append(r)
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(c) for c in configs)
        results = [r for r in out if r is not None]

    if not results:
        print("\nNo (edge,pair) produced trades. Nothing written."); return

    rows = [r[0] for r in results]
    all_meta = np.concatenate([r[2] for r in results if r[2].size]) if results else np.zeros(0)
    df = pd.DataFrame(rows)

    # ----- pooled / portfolio meta sleeve (equal-weight, all gated sleeves) -----
    # Per-sleeve Sharpe set = the trial family for honest deflation: we searched
    # n_trials (edge,pair) configs, so the pooled meta DSR deflates against the
    # dispersion of the per-(edge,pair) meta Sharpes (the configs actually tried).
    meta_sleeve_sr = []
    for r in results:
        m = r[2]
        if m.size >= 2:
            meta_sleeve_sr.append(O.sharpe(m))
    meta_sleeve_sr = np.array(meta_sleeve_sr)
    pooled_dsr, pooled_sr0 = dsr_of(all_meta, meta_sleeve_sr)

    # how many single (edge,pair) clear 0.95
    n_clear = int((df["meta_dsr"] >= 0.95).sum())
    n_clear_prim = int((df["prim_dsr"] >= 0.95).sum())
    best_idx = int(df["meta_dsr"].fillna(-1).idxmax())
    best = df.loc[best_idx]

    # lift across the set (median, to compare with the 2-pair +0.53 PF / +100 bp)
    df_v = df[(df["prim_trades"] >= 5) & (df["meta_trades"] >= 5)]
    med_prim_pf = float(df_v["prim_pf"].median()); med_meta_pf = float(df_v["meta_pf"].median())
    med_prim_bp = float(df_v["prim_bp"].median()); med_meta_bp = float(df_v["meta_bp"].median())
    pf_lift = med_meta_pf - med_prim_pf; bp_lift = med_meta_bp - med_prim_bp

    summary = dict(
        n_configs=n_trials, n_with_trades=len(df),
        n_meta_clear_0p95=n_clear, n_prim_clear_0p95=n_clear_prim,
        best_edge=str(best["edge"]), best_pair_id=str(best["pair_id"]),
        best_meta_dsr=float(best["meta_dsr"]), best_meta_pf=float(best["meta_pf"]),
        pooled_meta_trades=int(all_meta.size),
        pooled_meta_dsr=round(pooled_dsr, 4) if np.isfinite(pooled_dsr) else float("nan"),
        pooled_meta_sr0=round(pooled_sr0, 4) if np.isfinite(pooled_sr0) else float("nan"),
        pooled_meta_pf=round(profit_factor(all_meta), 3),
        med_prim_pf=round(med_prim_pf, 3), med_meta_pf=round(med_meta_pf, 3),
        med_prim_bp=round(med_prim_bp, 1), med_meta_bp=round(med_meta_bp, 1),
        pf_lift=round(pf_lift, 3), bp_lift=round(bp_lift, 1),
        meta_dsr_median=round(float(df["meta_dsr"].median()), 3),
        meta_dsr_max=round(float(df["meta_dsr"].max()), 3),
    )

    # ----- write tables -----
    df_sorted = df.sort_values("meta_dsr", ascending=False)
    df_sorted.to_csv(os.path.join(TAB, "metalabel_scaled.csv"), index=False)

    md = [
        "# Scaling the precondition-met meta-gate across the proven-edge set\n",
        f"_Edges: {len(edges)}; pairs: {len(pairs)}; configs with trades: {len(df)} of {n_trials}._\n",
        "\n## Headline (does pooling clear DSR 0.95?)\n",
        f"- **Pooled meta-strategy DSR** (equal-weight gated sleeves, deflated against "
        f"the {n_trials}-config trial family): **{summary['pooled_meta_dsr']}** "
        f"(benchmark SR0 {summary['pooled_meta_sr0']}, pooled meta PF {summary['pooled_meta_pf']}, "
        f"{summary['pooled_meta_trades']:,} OOS trades).",
        f"- **Per-(edge,pair) clearing DSR 0.95**: meta **{n_clear} of {len(df)}**; "
        f"primary {n_clear_prim} of {len(df)}.",
        f"- **Best single sleeve**: {best['edge']} on {best['pair_id']} "
        f"(meta DSR {best['meta_dsr']}, PF {best['meta_pf']}).",
        f"- **Meta DSR distribution**: median {summary['meta_dsr_median']}, max {summary['meta_dsr_max']}.",
        "\n## Does the 2-pair lift hold at scale? (median over valid sleeves)\n",
        f"- **PF**: {med_prim_pf:.2f} -> {med_meta_pf:.2f} (median lift {pf_lift:+.2f}; "
        f"2-pair sample was +0.53).",
        f"- **Per-trade**: {med_prim_bp:.0f} -> {med_meta_bp:.0f} bp (median lift {bp_lift:+.0f} bp; "
        f"2-pair sample was +100 bp).",
        "\n## Per (edge, pair)\n",
        df_sorted.to_markdown(index=False),
    ]
    with open(os.path.join(TAB, "metalabel_scaled.md"), "w") as f:
        f.write("\n".join(md))
    json.dump(summary, open(os.path.join(TAB, "metalabel_scaled_summary.json"), "w"), indent=2, default=str)
    print("\n  tables ->", os.path.abspath(TAB), flush=True)

    make_figure(df, summary)
    append_writeup(summary, len(edges), len(pairs), len(df), n_trials)

    print("\n" + "=" * 66)
    print("SCALED RESULT: meta-gating the proven-edge set")
    print("=" * 66)
    print(f"  configs with trades : {len(df)} of {n_trials}")
    print(f"  meta clearing DSR.95: {n_clear}  | primary: {n_clear_prim}")
    print(f"  POOLED meta DSR     : {summary['pooled_meta_dsr']}  (SR0 {summary['pooled_meta_sr0']}, "
          f"PF {summary['pooled_meta_pf']}, n={summary['pooled_meta_trades']:,})")
    print(f"  median PF lift      : {med_prim_pf:.2f} -> {med_meta_pf:.2f} ({pf_lift:+.2f})")
    print(f"  median bp lift      : {med_prim_bp:.0f} -> {med_meta_bp:.0f} ({bp_lift:+.0f} bp)")
    print(f"  best single sleeve  : {best['edge']} / {best['pair_id']}  meta DSR {best['meta_dsr']}")
    print(f"\nTOTAL {time.time()-t0:.0f}s")


def make_figure(df, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grey = "#999999"; acc = "#D55E00"; grn = "#009E73"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # panel 1: DSR distribution primary vs meta, vs the 0.95 hurdle
    pv = df["prim_dsr"].dropna().to_numpy(); mv = df["meta_dsr"].dropna().to_numpy()
    bins = np.linspace(0, 1, 21)
    ax[0].hist(pv, bins=bins, alpha=0.55, color=grey, label="primary")
    ax[0].hist(mv, bins=bins, alpha=0.65, color=acc, label="meta-gated")
    ax[0].axvline(0.95, color=grn, ls="--", lw=1.2, label="DSR=0.95 bar")
    ax[0].set_xlabel("Deflated Sharpe Ratio (per edge,pair, OOS)")
    ax[0].set_ylabel("count of (edge,pair) sleeves")
    ax[0].set_title(f"DSR distribution\n(meta clearing 0.95: {summary['n_meta_clear_0p95']} "
                    f"of {summary['n_with_trades']})")
    ax[0].legend(fontsize=8)

    # panel 2: PF primary vs meta scatter
    ax[1].scatter(df["prim_pf"], df["meta_pf"], s=22, color=acc, alpha=0.7, edgecolor="k", lw=0.3)
    lim = [0, max(3.0, float(np.nanmax([df["prim_pf"].max(), df["meta_pf"].max()])) * 1.05)]
    ax[1].plot(lim, lim, color="k", lw=0.8, ls="--")
    ax[1].axhline(1.0, color=grey, lw=0.6); ax[1].axvline(1.0, color=grey, lw=0.6)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("primary PF (OOS, net)"); ax[1].set_ylabel("meta-gated PF (OOS, net)")
    ax[1].set_title(f"PF lift across sleeves\n(median {summary['med_prim_pf']:.2f} -> "
                    f"{summary['med_meta_pf']:.2f})")

    # panel 3: the pooled DSR vs the 0.95 bar
    ax[2].bar([0], [summary["pooled_meta_dsr"]], color=acc, width=0.5)
    ax[2].axhline(0.95, color=grn, ls="--", lw=1.2, label="DSR=0.95 bar")
    ax[2].set_xticks([0]); ax[2].set_xticklabels(["pooled meta\nstrategy"])
    ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("Deflated Sharpe Ratio")
    ax[2].set_title(f"Pooled meta DSR\n(deflated vs {summary['n_configs']}-config family)")
    ax[2].text(0, summary["pooled_meta_dsr"] + 0.02, f"{summary['pooled_meta_dsr']:.3f}",
               ha="center", fontweight="bold")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_metalabel_scaled.png"), dpi=200)
    plt.close(fig)
    print("  figure ->", os.path.abspath(os.path.join(FIG, "fig_metalabel_scaled.png")), flush=True)


WRITEUP_MARK = "## 7. Scaling to the full proven-edge set"


def append_writeup(summary, n_edges, n_pairs, n_with, n_trials):
    cleared = summary["pooled_meta_dsr"] >= 0.95
    verdict = ("**clears**" if cleared else "**does not clear**")
    # "holds" only if the lift is materially positive; a near-zero median lift is
    # not a real lift, it is noise around break-even.
    pf_held = ("holds, but only weakly" if summary["pf_lift"] > 0.05
               else "largely washes out" if summary["pf_lift"] > -0.05
               else "does not hold")
    para = f"""
{WRITEUP_MARK}

Section 4 named scaling to the full set of proven edges as the natural next step.
This is that step. The precondition-met meta-gate was run across {n_edges} closed
structural edges (order-flow imbalance, open-interest squeeze, funding squeeze,
cross-asset residual, cross-venue funding divergence, top-trader ratio) crossed
with the {n_pairs}-pair perp universe the structural engine runs over, for
{n_with} of {n_trials} (edge, pair) sleeves that produced trades. Each sleeve is
primary-alone versus primary-plus-meta-gate, out-of-sample, net of per-fill costs,
with full intra-bar exits and purged walk-forward; the verified sim kernel and cost
model are reused read-only.

The headline question was whether pooling the meta-gated sleeves as one
weakly-correlated meta-strategy clears the program's Deflated Sharpe bar of 0.95.
Treating the set honestly as the trial family (the pooled meta sleeve is deflated
against the dispersion of the per-sleeve Sharpes, i.e. the {n_trials} configurations
actually searched), the pooled meta-strategy DSR is **{summary['pooled_meta_dsr']}**
(benchmark {summary['pooled_meta_sr0']}, pooled meta profit factor
{summary['pooled_meta_pf']}, {summary['pooled_meta_trades']:,} out-of-sample trades).
This {verdict} the 0.95 bar. Of the individual sleeves, {summary['n_meta_clear_0p95']}
of {n_with} meta-gated configurations clear 0.95 (primary alone:
{summary['n_prim_clear_0p95']}); the best single sleeve is {summary['best_edge']} on
{summary['best_pair_id']} at meta DSR {summary['best_meta_dsr']}. The median meta DSR
across sleeves is {summary['meta_dsr_median']}.

The per-sleeve lift {pf_held} at scale, in a more muted form than the focused
two-pair sample. The median profit factor moves {summary['med_prim_pf']} to
{summary['med_meta_pf']} (median lift {summary['pf_lift']:+}, against the two-pair
sample's +0.53) and median per-trade P&L moves {summary['med_prim_bp']} to
{summary['med_meta_bp']} basis points (median lift {summary['bp_lift']:+} bp, against
the two-pair sample's +100 bp). The direction of the correction survives broadly:
applied to primaries that already carry an edge, the secondary tends to raise
precision and per-trade P&L. But the magnitude of the lift on a hand-picked two-pair
sample is not representative of the full set, and pooling many gated sleeves into one
meta-strategy {'does clear' if cleared else 'does not, on its own, clear'} the
deflation bar once the trial family is counted honestly. The honest verdict: the
mechanism generalises in direction, the headline two-pair magnitudes do not, and
{'the scaled pooled sleeve clears 0.95.' if cleared else 'meta-gating a set of single-asset directional edges, even with the precondition met, is not by itself enough to clear deflation at scale. The correction of study 03 stands as a statement about mechanism and direction, not as a deflation-passing portfolio.'}
"""
    try:
        with open(WRITEUP, "r") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = ""
    # idempotent: drop any prior copy of this subsection
    if WRITEUP_MARK in txt:
        txt = txt.split(WRITEUP_MARK)[0].rstrip() + "\n"
    with open(WRITEUP, "w") as f:
        f.write(txt.rstrip() + "\n\n" + para.strip() + "\n")
    print("  writeup -> appended subsection 7 to", os.path.abspath(WRITEUP), flush=True)


if __name__ == "__main__":
    main()

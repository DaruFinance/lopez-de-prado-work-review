#!/usr/bin/env python3
"""
Project 1 (extension), Imbalance bars and run bars, the advanced tier of LdP's
information-driven bars (Advances in Financial Machine Learning, Ch. 2).

The existing study covers time / tick / volume / dollar bars and shows the
information-driven ones Gaussianize returns. This extension adds the next tier:

  * TICK-IMBALANCE bars (TIB)   : sample when accumulated SIGNED tick flow
                                  breaks an adaptive expectation.
  * VOLUME-IMBALANCE bars (VIB) : same idea on SIGNED traded volume.
  * TICK-RUN bars (TRB)         : sample when a one-sided RUN of signed ticks
                                  breaks its expectation.
  * VOLUME-RUN bars (VRB)       : same on signed volume.

All four sample when the imbalance / run exceeds an EWMA-tracked expected
threshold, exactly the recipe in AFML Ch. 2 (eqs. 2.3-2.10). The threshold is
built only from information available up to the current base bar, so bar
construction is strictly causal (no look-ahead in the adaptive trigger).

Tick sign uses the tick rule (sign of close-to-close change, ties carry the
prior sign). The signed-volume split uses each 1m bar's taker-buy / taker-sell
notional, which Binance reports directly, so signed volume is observed, not
inferred.

Question: does sampling on order-flow IMBALANCE thin the tails further than
dollar / volume / tick bars? And is any benefit granularity-dependent, like the
plain information-bar result already documented for this study?

Outputs:
  tables/imbalance_run_bars.{csv,md}
  figures/fig_imbalance_run_bars.png
  (plus a writeup subsection appended by hand from these numbers)

Real data only. Light footprint: a handful of representative instruments, one
1m series held in memory at a time. Peak RAM stays well under 2 GB.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.exists(_os.path.join(_d, "config.py")):
    _d = _os.path.dirname(_d)
REPO_ROOT = _d
_sys.path.insert(0, REPO_ROOT)
import config as cfg
from config import LIB as _LIB
_sys.path.insert(0, _LIB)
import bars as B
import barstats as S
import style as ST

warnings.filterwarnings("ignore")
ST.set_style()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 1m base bars (<PAIR>_1m.parquet) come from the repo data config; see DATA.md.
CACHE = cfg.CRYPTO_1M

# Representative instruments: the study's headline pair plus two majors. Kept
# small on purpose (this is a distributional-property check, not a panel).
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Existing study numbers (median excess kurtosis, clean 1m, ~daily target) for
# the side-by-side reference column in the markdown table.
EXISTING_EXKURT = {"time": 5.29, "tick": 1.61, "volume": 1.48, "dollar": 1.54}

NEW_TYPES = ["tib", "vib", "trb", "vrb"]
PRETTY = {"tib": "tick-imbalance", "vib": "volume-imbalance",
          "trb": "tick-run", "vrb": "volume-run"}


def load_1m(path):
    df = pd.read_parquet(path)
    df = df.set_index(pd.to_datetime(df["open_time"], utc=True)).drop(columns=["open_time"])
    return df[(df["close"] > 0) & (df["volume"] > 0) & (df["count"] > 0)]


def tick_signs(close):
    """AFML tick rule: b_t = sign(dp_t), ties carry the previous sign (b_0=+1)."""
    dp = np.diff(close, prepend=close[0])
    b = np.sign(dp)
    # carry prior sign across zero-change bars
    b = b.astype(np.float64)
    last = 1.0
    out = np.empty_like(b)
    for i in range(b.shape[0]):
        if b[i] == 0.0:
            out[i] = last
        else:
            out[i] = b[i]
            last = b[i]
    return out


try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (a and callable(a[0])) is False else a[0]


@njit(cache=True)
def _imbalance_gid(b, w, alpha_T, alpha_v, warmup, ewma_T0):
    """Causal tick/volume IMBALANCE-bar grouping (AFML eq. 2.4-2.7).

    b : signed flow per base bar  (b_t for ticks, b_t*v_t for signed volume).
    w : the magnitude clock per base bar (1.0 for ticks, v_t for volume), so the
        bar SIZE accumulates in the same w-units as the expected-size seed E[T].

    A bar closes when the cumulative SIGNED flow breaks its expectation:
        |theta| >= E[T] * |E[b_per_unit]|
    where theta = sum of signed flow in the open bar, E[T] = expected bar size
    (in w-units) and E[b_per_unit] = expected signed flow per w-unit. Both sides
    are in flow units, which is the AFML imbalance trigger (eq. 2.6). Comparing
    |theta| (not the raw size) is what makes these IMBALANCE bars: a balanced
    stretch of flow keeps theta near zero and the bar stays open; a one-sided
    burst closes it quickly.

    EWMAs:
      E[T]          = expected bar size in w-units. It is the FREQUENCY target
                      (matched to the threshold bars) and is held fixed at the
                      seed ewma_T0. Adapting it to the realized imbalance-bar
                      length is unstable: a short bar shrinks E[T], which shrinks
                      the trigger, which shortens the next bar, collapsing every
                      base bar into its own bar. Fixing E[T] removes that
                      feedback while leaving the trigger causal.
      E[b_per_unit] = EWMA of (theta_bar / size_bar) over closed bars; updates
                      only AFTER a bar closes, so the trigger for bar k uses only
                      bars 0..k-1 -> strictly causal.
    During warmup (first `warmup` closed bars) the expected per-unit imbalance is
    not yet meaningful, so we close on the size clock (size >= E[T]) to seed the
    frequency, exactly the matched-frequency convention the threshold bars use. A
    safety cap (size >= cap_mult * E[T]) forces a close in a persistently
    balanced stretch so no single bar can run away. Returns a non-decreasing
    int64 group id per base bar.
    """
    n = b.shape[0]
    gid = np.empty(n, np.int64)
    g = 0
    theta = 0.0
    size = 0.0
    exp_T = ewma_T0                      # FIXED frequency target (not adapted)
    exp_b = 0.0
    have_b = False
    nbars = 0
    cap_mult = 50.0                      # hard size ceiling, in multiples of E[T]
    for i in range(n):
        theta += b[i]
        size += w[i]
        gid[i] = g
        close = False
        if nbars < warmup or not have_b:
            # frequency-seeding phase: close on the size clock
            if size >= exp_T and size > 0.0:
                close = True
        else:
            trigger = exp_T * abs(exp_b)
            if trigger > 0.0 and abs(theta) >= trigger:
                close = True
            elif size >= cap_mult * exp_T and size > 0.0:
                # balanced-flow safety cap: never let a bar run forever
                close = True
        if close:
            this_b_per_unit = theta / size
            if have_b:
                exp_b = alpha_v * this_b_per_unit + (1.0 - alpha_v) * exp_b
            else:
                exp_b = this_b_per_unit
                have_b = True
            theta = 0.0
            size = 0.0
            nbars += 1
            g += 1
    # tail bar (open partial) keeps its gid; _aggregate will emit it
    return gid


@njit(cache=True)
def _run_gid(b, w, alpha_T, alpha_v, warmup, ewma_T0):
    """Causal tick/volume RUN-bar grouping (AFML eq. 2.8-2.10).

    Tracks the larger of the cumulative BUY-side and SELL-side flow within the
    current bar (a one-sided 'run' of order flow). A bar closes when:
        max(theta_buy, theta_sell) >= E[T] * max(P_buy*E[w|+], P_sell*E[w|-])
    Implemented with EWMA-tracked expected proportions, mirroring the imbalance
    builder. Same causality contract (EWMAs update only after a bar closes).
    """
    n = b.shape[0]
    gid = np.empty(n, np.int64)
    g = 0
    theta_buy = 0.0
    theta_sell = 0.0
    size = 0.0
    exp_T = ewma_T0
    exp_pos = 0.5              # EWMA of (buy-side w-share)
    have = False
    nbars = 0
    for i in range(n):
        if b[i] > 0.0:
            theta_buy += w[i]
        elif b[i] < 0.0:
            theta_sell += w[i]
        size += w[i]
        gid[i] = g
        if nbars < warmup or not have:
            trigger = exp_T * 0.5
        else:
            share = exp_pos if exp_pos > (1.0 - exp_pos) else (1.0 - exp_pos)
            trigger = exp_T * share
        if trigger <= 0.0:
            trigger = exp_T * 0.5
        run = theta_buy if theta_buy > theta_sell else theta_sell
        if run >= trigger and size > 0.0:
            pos_share = theta_buy / size
            exp_T = alpha_T * size + (1.0 - alpha_T) * exp_T
            if have:
                exp_pos = alpha_v * pos_share + (1.0 - alpha_v) * exp_pos
            else:
                exp_pos = pos_share
                have = True
            theta_buy = 0.0
            theta_sell = 0.0
            size = 0.0
            nbars += 1
            g += 1
    return gid


def build_new_bars(df, n_target):
    """Build the four imbalance/run bar types, each tuned to ~n_target bars.

    The EWMA seed E[T] (initial expected bar size) sets the average frequency,
    so we seed it from total-activity / n_target, the same matched-frequency
    convention the study uses for the threshold bars. alpha controls how fast
    the expectation adapts (AFML uses a slow EWMA; 2/(L+1) with L a few hundred
    base bars). Everything here is causal.
    """
    close = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    bsign = tick_signs(close)                          # +/-1 per base bar
    # signed volume from observed taker buy/sell notional split:
    # net signed share of the bar's volume = (buy - sell) / total, in [-1, 1].
    qv = df["quote_volume"].to_numpy(np.float64)
    bqv = df["taker_buy_quote_volume"].to_numpy(np.float64)
    sqv = qv - bqv
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_share = np.where(qv > 0, (bqv - sqv) / qv, 0.0)
    signed_vol = signed_share * vol                    # b_t * v_t, signed volume

    n = len(df)
    alpha_T = 2.0 / (200.0 + 1.0)                      # ~200-bar EWMA of bar size
    alpha_v = 2.0 / (200.0 + 1.0)
    warmup = max(20, n_target // 20)

    seed_T_ticks = max(1.0, n / max(1, n_target))      # E[T] in tick (base-bar) units
    seed_T_vol = max(1e-9, vol.sum() / max(1, n_target))  # E[T] in volume units

    ones = np.ones(n, np.float64)

    gid_tib = _imbalance_gid(bsign.astype(np.float64), ones,
                             alpha_T, alpha_v, warmup, seed_T_ticks)
    gid_vib = _imbalance_gid(signed_vol, vol,
                             alpha_T, alpha_v, warmup, seed_T_vol)
    gid_trb = _run_gid(bsign.astype(np.float64), ones,
                       alpha_T, alpha_v, warmup, seed_T_ticks)
    gid_vrb = _run_gid(np.sign(signed_vol), vol,
                       alpha_T, alpha_v, warmup, seed_T_vol)

    return {
        "tib": B._aggregate(df, gid_tib),
        "vib": B._aggregate(df, gid_vib),
        "trb": B._aggregate(df, gid_trb),
        "vrb": B._aggregate(df, gid_vrb),
    }


def process(name):
    path = os.path.join(CACHE, f"{name}_1m.parquet")
    df = load_1m(path)
    days = max(50, int((df.index[-1] - df.index[0]).days))
    # existing study bar types at the same ~daily target, for an in-run baseline
    base_set = B.matched_bars(df, n_target=days)
    new_set = build_new_bars(df, n_target=days)
    rows, qq = [], {}
    for bt, bars in {**base_set, **new_set}.items():
        st = S.return_stats(B.log_returns(bars))
        rows.append(dict(pair=name, bar_type=bt, n_bars=len(bars),
                         **{k: st[k] for k in ["n", "skew", "exkurt", "jb",
                                               "jb_p", "ac1", "lb_p"]}))
        qq[bt] = st.get("std_returns")
    del df
    return pd.DataFrame(rows), (name, qq)


def _imbalance_gid_ref(b, w, alpha_T, alpha_v, warmup, ewma_T0):
    """Pure-Python mirror of _imbalance_gid for correctness verification."""
    n = b.shape[0]
    gid = np.empty(n, np.int64)
    g = 0; theta = 0.0; size = 0.0; exp_T = ewma_T0; exp_b = 0.0
    have_b = False; nbars = 0; cap_mult = 50.0
    for i in range(n):
        theta += b[i]; size += w[i]; gid[i] = g
        close = False
        if nbars < warmup or not have_b:
            if size >= exp_T and size > 0.0:
                close = True
        else:
            trigger = exp_T * abs(exp_b)
            if trigger > 0.0 and abs(theta) >= trigger:
                close = True
            elif size >= cap_mult * exp_T and size > 0.0:
                close = True
        if close:
            this_b_per_unit = theta / size
            if have_b:
                exp_b = alpha_v * this_b_per_unit + (1.0 - alpha_v) * exp_b
            else:
                exp_b = this_b_per_unit; have_b = True
            theta = 0.0; size = 0.0; nbars += 1; g += 1
    return gid


def _run_gid_ref(b, w, alpha_T, alpha_v, warmup, ewma_T0):
    """Pure-Python mirror of _run_gid for correctness verification."""
    n = b.shape[0]
    gid = np.empty(n, np.int64)
    g = 0; theta_buy = 0.0; theta_sell = 0.0; size = 0.0
    exp_T = ewma_T0; exp_pos = 0.5; have = False; nbars = 0
    for i in range(n):
        if b[i] > 0.0:
            theta_buy += w[i]
        elif b[i] < 0.0:
            theta_sell += w[i]
        size += w[i]; gid[i] = g
        if nbars < warmup or not have:
            trigger = exp_T * 0.5
        else:
            share = exp_pos if exp_pos > (1.0 - exp_pos) else (1.0 - exp_pos)
            trigger = exp_T * share
        if trigger <= 0.0:
            trigger = exp_T * 0.5
        run = theta_buy if theta_buy > theta_sell else theta_sell
        if run >= trigger and size > 0.0:
            pos_share = theta_buy / size
            exp_T = alpha_T * size + (1.0 - alpha_T) * exp_T
            if have:
                exp_pos = alpha_v * pos_share + (1.0 - alpha_v) * exp_pos
            else:
                exp_pos = pos_share; have = True
            theta_buy = 0.0; theta_sell = 0.0; size = 0.0; nbars += 1; g += 1
    return gid


def verify_kernels():
    """Bit-identical check: Numba kernels vs pure-Python reference on a slice."""
    rng = np.random.default_rng(0)
    n = 4000
    b = rng.choice([-1.0, 1.0], size=n)
    w = rng.gamma(2.0, 1.0, size=n)
    aT, av, wu, seed = 2.0 / 201.0, 2.0 / 201.0, 50, 25.0
    deltas = []
    g1 = _imbalance_gid(b, w, aT, av, wu, seed)
    g1r = _imbalance_gid_ref(b, w, aT, av, wu, seed)
    deltas.append(int(np.max(np.abs(g1 - g1r))))
    g2 = _imbalance_gid(b, np.ones(n), aT, av, wu, float(n) / 200.0)
    g2r = _imbalance_gid_ref(b, np.ones(n), aT, av, wu, float(n) / 200.0)
    deltas.append(int(np.max(np.abs(g2 - g2r))))
    g3 = _run_gid(b, w, aT, av, wu, seed)
    g3r = _run_gid_ref(b, w, aT, av, wu, seed)
    deltas.append(int(np.max(np.abs(g3 - g3r))))
    g4 = _run_gid(b, np.ones(n), aT, av, wu, float(n) / 200.0)
    g4r = _run_gid_ref(b, np.ones(n), aT, av, wu, float(n) / 200.0)
    deltas.append(int(np.max(np.abs(g4 - g4r))))
    md = max(deltas)
    print(f"[verify] numba vs pure-python max|delta| over 4 kernels = {md} "
          f"(per-kernel {deltas})")
    if md != 0:
        raise SystemExit("Numba kernel does NOT match pure-Python reference.")
    return md


def main():
    verify_kernels()
    avail = [p for p in PAIRS
             if os.path.exists(os.path.join(CACHE, f"{p}_1m.parquet"))]
    if not avail:
        print("No 1m cache found at the expected location."); sys.exit(1)
    print(f"instruments: {avail}")

    per_pair, qqs = [], {}
    for name in avail:
        pp, (n, q) = process(name)
        per_pair.append(pp)
        qqs[n] = q
        sub = pp.set_index("bar_type")["exkurt"]
        print(f"  {name}: " + "  ".join(f"{bt}={sub.get(bt, float('nan')):.2f}"
              for bt in ["time", "tick", "volume", "dollar", *NEW_TYPES]))
    per_pair = pd.concat(per_pair, ignore_index=True)

    order = ["time", "tick", "volume", "dollar", *NEW_TYPES]
    per_pair.to_csv(f"{PROJ}/tables/imbalance_run_bars.csv", index=False)

    agg = (per_pair.assign(abs_skew=per_pair["skew"].abs(),
                           abs_ac1=per_pair["ac1"].abs())
           .groupby("bar_type")
           .agg(pairs=("pair", "nunique"),
                med_n_bars=("n_bars", "median"),
                med_abs_skew=("abs_skew", "median"),
                med_exkurt=("exkurt", "median"),
                med_abs_ac1=("abs_ac1", "median"),
                med_jb=("jb", "median"),
                frac_normal=("jb_p", lambda s: float((s > 0.05).mean())))
           .reindex(order))

    with open(f"{PROJ}/tables/imbalance_run_bars.md", "w") as fh:
        fh.write("# Imbalance and run bars vs the existing bar types "
                 f"({len(avail)} Binance perps, 1m base, ~daily target)\n\n")
        fh.write("Median across instruments. Lower |skew|, excess kurtosis, "
                 "|AC(1)|, Jarque-Bera are closer to IID-Gaussian; higher "
                 "frac_normal is better. The four new rows (tib / vib / trb / "
                 "vrb) are the advanced LdP bars; the first four rows are the "
                 "study's existing bars rebuilt on the same instruments.\n\n")
        fh.write(agg.round(4).to_markdown())
        fh.write("\n\n## Reference: cross-section medians from the main study "
                 "(27 perps)\n\n")
        ref = pd.Series(EXISTING_EXKURT, name="med_exkurt (27-perp study)")
        fh.write(ref.to_frame().round(2).to_markdown())
        fh.write("\n\n*tib = tick-imbalance, vib = volume-imbalance, "
                 "trb = tick-run, vrb = volume-run.*\n")

    print("\n=== MEDIAN ACROSS INSTRUMENTS ===")
    print(agg.round(4).to_string())

    make_fig(per_pair, qqs, order, avail)
    print("\nImbalance/run-bar extension done.")


def make_fig(per_pair, qqs, order, avail):
    figd = f"{PROJ}/figures"
    colors = {"time": ST.barcolor("time"), "tick": ST.barcolor("tick"),
              "volume": ST.barcolor("volume"), "dollar": ST.barcolor("dollar"),
              "tib": "#CC79A7", "vib": "#882255", "trb": "#0072B2", "vrb": "#117733"}
    labels = {"time": "time", "tick": "tick", "volume": "volume",
              "dollar": "dollar", "tib": "tick-imbal", "vib": "vol-imbal",
              "trb": "tick-run", "vrb": "vol-run"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: median excess kurtosis by bar type (existing + new).
    med = per_pair.groupby("bar_type")["exkurt"].median().reindex(order)
    xs = np.arange(len(order))
    bars = axes[0].bar(xs, med.values, color=[colors[b] for b in order], alpha=0.85)
    axes[0].axhline(0, color="black", lw=0.9, ls="--", alpha=0.6)
    for x, v in zip(xs, med.values):
        axes[0].text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([labels[b] for b in order], rotation=30, ha="right")
    axes[0].set_ylabel("Median excess kurtosis of bar returns")
    axes[0].set_title("Excess kurtosis by bar type\n0 = Gaussian; lower is better")
    # vertical divider between existing and new bars
    axes[0].axvline(3.5, color="#888888", lw=1, ls=":")
    axes[0].text(1.5, axes[0].get_ylim()[1] * 0.92, "existing", ha="center",
                 fontsize=9, color="#555555")
    axes[0].text(5.5, axes[0].get_ylim()[1] * 0.92, "imbalance / run",
                 ha="center", fontsize=9, color="#555555")

    # Right: QQ overlay for the headline pair (dollar vs the two imbalance bars).
    name = "BTCUSDT" if "BTCUSDT" in qqs else avail[0]
    for bt in ["time", "dollar", "vib", "vrb"]:
        sr = qqs[name].get(bt)
        if sr is None or len(sr) < 50:
            continue
        sr = np.sort(sr.to_numpy())
        q = ss.norm.ppf((np.arange(1, len(sr) + 1) - 0.5) / len(sr))
        axes[1].plot(q, sr, ".", ms=2.6, color=colors[bt], label=labels[bt], alpha=0.7)
    axes[1].plot([-5, 5], [-5, 5], "k--", lw=0.9, alpha=0.7)
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-8, 8)
    axes[1].set_xlabel("Theoretical Gaussian quantiles")
    axes[1].set_ylabel("Standardized return quantiles")
    axes[1].set_title(f"Normal QQ overlay, {name}\ncloser to the dashed line = more Gaussian")
    axes[1].legend(markerscale=4)

    fig.suptitle("Advanced bars: do imbalance and run bars thin the tails further?",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{figd}/fig_imbalance_run_bars.png"); plt.close(fig)
    print("figure written:", f"{figd}/fig_imbalance_run_bars.png")


if __name__ == "__main__":
    main()

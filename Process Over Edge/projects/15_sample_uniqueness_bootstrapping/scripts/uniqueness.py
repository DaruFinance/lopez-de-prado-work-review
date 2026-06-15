"""
uniqueness.py: Sample Uniqueness & Sequential Bootstrap (Lopez de Prado, AFML Ch.4).

The non-IID-labels problem. A triple-barrier label spanning bars [t0, t1] shares
information with every other label whose span overlaps it, so the observations
are NOT independent. This module reproduces LdP's apparatus:

  1. CONCURRENCY  c_t  = number of labels whose [t0, t1] span is live at bar t.
  2. AVERAGE UNIQUENESS  u_i = mean over label i's span of 1 / c_t.
       Average uniqueness << 1 means overlap is severe.
  3. EFFECTIVE SAMPLE SIZE  = sum_i u_i  <<  N  (naive IID counting overstates
       the information content of the sample).
  4. SEQUENTIAL BOOTSTRAP  (AFML Snippet 4.5/4.6): draw samples with probability
       favouring low overlap with the already-drawn set, so a bootstrap sample
       has materially HIGHER average uniqueness than a standard IID bootstrap.
  5. SAMPLE WEIGHTS: (a) uniqueness weights u_i; (b) return-attribution weights
       w_i proportional to the sum over the span of |log-return_t| / c_t.

Hot loops (concurrency, average uniqueness, the per-draw uniqueness scan in the
sequential bootstrap) are Numba @njit. A pure-Python reference verifies each
kernel is bit-identical (max|delta| reported by verify.py).

Scalability: we never materialise a dense T x N indicator matrix (that is the
RAM bomb LdP's textbook snippet builds). The label set is stored as integer
spans (t0, t1); concurrency is an O(N + T) running array; the sequential
bootstrap maintains an O(T) running concurrency-contribution vector and scores
candidates in O(N * avg_span) per draw. This scales to the full cross-section.
"""
from __future__ import annotations
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (a and callable(a[0])) else a[0]


# --------------------------------------------------------------------------- #
# 1. Concurrency  c_t = # labels live at bar t
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _concurrency_kernel(t0, t1, n_bars):
    """Concurrency per bar via a difference array. c_t counts labels whose span
    [t0_i, t1_i] (inclusive) covers bar t. O(N + T)."""
    n = t0.shape[0]
    diff = np.zeros(n_bars + 1, np.int64)
    for i in range(n):
        a = t0[i]
        b = t1[i]
        diff[a] += 1
        diff[b + 1] -= 1
    c = np.empty(n_bars, np.int64)
    run = 0
    for t in range(n_bars):
        run += diff[t]
        c[t] = run
    return c


def concurrency(t0, t1, n_bars):
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    return _concurrency_kernel(t0, t1, int(n_bars))


def concurrency_reference(t0, t1, n_bars):
    """Pure-Python reference: increment every bar in every span."""
    c = np.zeros(int(n_bars), dtype=np.int64)
    for i in range(len(t0)):
        for t in range(int(t0[i]), int(t1[i]) + 1):
            c[t] += 1
    return c


# --------------------------------------------------------------------------- #
# 2. Average uniqueness  u_i = mean_{t in span_i} 1 / c_t
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _avg_uniqueness_kernel(t0, t1, c):
    """Average uniqueness per label from a precomputed concurrency array c."""
    n = t0.shape[0]
    u = np.empty(n, np.float64)
    for i in range(n):
        a = t0[i]
        b = t1[i]
        s = 0.0
        m = 0
        for t in range(a, b + 1):
            ct = c[t]
            if ct > 0:
                s += 1.0 / ct
            m += 1
        u[i] = s / m if m > 0 else 0.0
    return u


def avg_uniqueness(t0, t1, c):
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    c = np.ascontiguousarray(np.asarray(c, np.int64))
    return _avg_uniqueness_kernel(t0, t1, c)


def avg_uniqueness_reference(t0, t1, c):
    n = len(t0)
    u = np.empty(n, np.float64)
    for i in range(n):
        span = list(range(int(t0[i]), int(t1[i]) + 1))
        vals = [1.0 / c[t] for t in span if c[t] > 0]
        u[i] = (sum(vals) / len(span)) if span else 0.0
    return u


# --------------------------------------------------------------------------- #
# 3. Sequential bootstrap (AFML Snippet 4.5 / 4.6)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _seq_bootstrap_kernel(t0, t1, n_bars, n_draw, rand_u):
    """Sequential bootstrap draw.

    Maintains `conc`, the running concurrency contributed by the already-drawn
    set. For each draw we compute, for every candidate i, its average uniqueness
    *as if it were added* :  mean_{t in span_i} 1 / (conc_t + 1).  Probabilities
    are normalised average-uniqueness; we sample one candidate by inverse-CDF
    using a precomputed uniform `rand_u[draw]`. Then we add the chosen label's
    span into `conc`. With-replacement (LdP's procedure).

    Returns the drawn indices (length n_draw)."""
    n = t0.shape[0]
    conc = np.zeros(n_bars, np.int64)
    out = np.empty(n_draw, np.int64)
    avgu = np.empty(n, np.float64)
    for d in range(n_draw):
        tot = 0.0
        for i in range(n):
            a = t0[i]
            b = t1[i]
            s = 0.0
            m = b - a + 1
            for t in range(a, b + 1):
                s += 1.0 / (conc[t] + 1.0)
            ui = s / m
            avgu[i] = ui
            tot += ui
        # inverse-CDF sample
        target = rand_u[d] * tot
        cum = 0.0
        chosen = n - 1
        for i in range(n):
            cum += avgu[i]
            if cum >= target:
                chosen = i
                break
        out[d] = chosen
        a = t0[chosen]
        b = t1[chosen]
        for t in range(a, b + 1):
            conc[t] += 1
    return out


@njit(cache=True)
def _seq_bootstrap_fast(t0, t1, n_bars, n_draw, rand_u,
                        bar_ptr, bar_lab):
    """Incremental sequential bootstrap. Mathematically identical to
    _seq_bootstrap_kernel but O(draw * span * labels_per_bar) instead of
    O(draw * N * span): only candidates whose spans intersect the bars of the
    just-added label have their average-uniqueness re-derived.

    bar_ptr/bar_lab is a CSR-style index: candidates covering bar t are
    bar_lab[bar_ptr[t] : bar_ptr[t+1]]. avgu[i] is kept current at all times.
    The selection (inverse-CDF over the running total) and the draw sequence are
    bit-identical to the reference because the same arithmetic is performed."""
    n = t0.shape[0]
    conc = np.zeros(n_bars, np.int64)
    # span sum S_i = sum_{t in span_i} 1/(conc_t+1); avgu_i = S_i / span_len_i
    S = np.empty(n, np.float64)
    span_len = np.empty(n, np.float64)
    avgu = np.empty(n, np.float64)
    tot = 0.0
    for i in range(n):
        m = t1[i] - t0[i] + 1
        span_len[i] = m
        S[i] = m * 1.0            # conc all zero -> each term = 1/(0+1) = 1
        avgu[i] = S[i] / m
        tot += avgu[i]
    out = np.empty(n_draw, np.int64)
    for d in range(n_draw):
        target = rand_u[d] * tot
        cum = 0.0
        chosen = n - 1
        for i in range(n):
            cum += avgu[i]
            if cum >= target:
                chosen = i
                break
        out[d] = chosen
        a = t0[chosen]
        b = t1[chosen]
        # adding the chosen span: each bar t in [a,b] goes conc -> conc+1, so
        # every candidate covering t loses (1/(conc_t+1) - 1/(conc_t+2)) from S.
        for t in range(a, b + 1):
            old = 1.0 / (conc[t] + 1.0)
            conc[t] += 1
            new = 1.0 / (conc[t] + 1.0)
            delta = old - new
            for p in range(bar_ptr[t], bar_ptr[t + 1]):
                j = bar_lab[p]
                tot -= avgu[j]
                S[j] -= delta
                avgu[j] = S[j] / span_len[j]
                tot += avgu[j]
    return out


def _build_bar_index(t0, t1, n_bars):
    """CSR index: for each bar t, the labels whose span covers t."""
    counts = np.zeros(n_bars + 1, np.int64)
    for i in range(len(t0)):
        for t in range(t0[i], t1[i] + 1):
            counts[t] += 1
    ptr = np.zeros(n_bars + 1, np.int64)
    acc = 0
    for t in range(n_bars):
        ptr[t] = acc
        acc += counts[t]
    ptr[n_bars] = acc
    bar_lab = np.empty(acc, np.int64)
    cur = ptr[:n_bars].copy()
    for i in range(len(t0)):
        for t in range(t0[i], t1[i] + 1):
            bar_lab[cur[t]] = i
            cur[t] += 1
    return ptr, bar_lab


def seq_bootstrap(t0, t1, n_bars, n_draw, rng, fast=True):
    """Sequential bootstrap of `n_draw` labels (with replacement).

    fast=True uses the incremental kernel (default; identical draws). The total
    bar-index memory is O(sum of span lengths) integers, far below a dense T x N
    indicator matrix."""
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    rand_u = rng.random(n_draw)
    if fast and len(t0) > 0:
        ptr, bar_lab = _build_bar_index(t0, t1, int(n_bars))
        return _seq_bootstrap_fast(t0, t1, int(n_bars), int(n_draw), rand_u,
                                   ptr, bar_lab)
    return _seq_bootstrap_kernel(t0, t1, int(n_bars), int(n_draw), rand_u)


def seq_bootstrap_reference(t0, t1, n_bars, n_draw, rng):
    """Pure-Python reference for the sequential bootstrap (same rand stream)."""
    n = len(t0)
    conc = np.zeros(int(n_bars), dtype=np.int64)
    out = np.empty(n_draw, dtype=np.int64)
    rand_u = rng.random(n_draw)
    for d in range(n_draw):
        avgu = np.empty(n)
        for i in range(n):
            a, b = int(t0[i]), int(t1[i])
            m = b - a + 1
            s = sum(1.0 / (conc[t] + 1.0) for t in range(a, b + 1))
            avgu[i] = s / m
        tot = avgu.sum()
        target = rand_u[d] * tot
        cum = 0.0
        chosen = n - 1
        for i in range(n):
            cum += avgu[i]
            if cum >= target:
                chosen = i
                break
        out[d] = chosen
        a, b = int(t0[chosen]), int(t1[chosen])
        for t in range(a, b + 1):
            conc[t] += 1
    return out


def standard_bootstrap(n, n_draw, rng):
    """Standard IID bootstrap: uniform draws with replacement."""
    return rng.integers(0, n, size=n_draw)


def sample_avg_uniqueness(sample_idx, t0, t1, c):
    """Average uniqueness of a drawn sample, evaluated against the FIXED full-set
    concurrency c (LdP measures a bootstrap sample's uniqueness against the
    concurrency of the labels actually present). For comparing standard vs
    sequential bootstraps we report the mean over the drawn indices of u_i."""
    u = avg_uniqueness(t0, t1, c)
    return float(np.mean(u[sample_idx]))


# --------------------------------------------------------------------------- #
# CUSUM symmetric filter (AFML Snippet 2.4): dense, event-based sampling.
# This is LdP's canonical event sampler: it fires whenever the cumulative
# absolute log-return since the last event crosses a threshold h. With a
# vertical barrier of max_hold bars, the resulting [t0, t1] spans overlap
# heavily (the non-IID-labels setting Ch.4 is about).
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cusum_kernel(log_ret, h):
    n = log_ret.shape[0]
    out = np.empty(n, np.int64)
    k = 0
    s_pos = 0.0
    s_neg = 0.0
    for t in range(1, n):
        r = log_ret[t]
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            s_pos = 0.0
            out[k] = t; k += 1
        elif s_neg <= -h:
            s_neg = 0.0
            out[k] = t; k += 1
    return out[:k]


def cusum_events(log_ret, h):
    """CUSUM symmetric-filter event bar indices (threshold h on |cum log-ret|)."""
    return _cusum_kernel(np.ascontiguousarray(np.asarray(log_ret, np.float64)), float(h))


def cusum_events_reference(log_ret, h):
    out = []
    s_pos = s_neg = 0.0
    for t in range(1, len(log_ret)):
        r = log_ret[t]
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            s_pos = 0.0; out.append(t)
        elif s_neg <= -h:
            s_neg = 0.0; out.append(t)
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 4. Return-attribution sample weights (AFML Snippet 4.10)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _return_attribution_kernel(t0, t1, c, log_ret):
    """Weight_i proportional to | sum_{t in span_i} log_ret_t / c_t |.
    log_ret_t is the bar log-return; attributing 1/c_t of each bar's return to
    each concurrent label de-duplicates overlapping information."""
    n = t0.shape[0]
    w = np.empty(n, np.float64)
    for i in range(n):
        a = t0[i]
        b = t1[i]
        s = 0.0
        for t in range(a, b + 1):
            ct = c[t]
            if ct > 0:
                s += log_ret[t] / ct
        w[i] = abs(s)
    return w


def return_attribution_weights(t0, t1, c, log_ret):
    """Return-attribution weights, normalised to sum to N (LdP convention:
    weights average to 1 so they are comparable to unit IID weights)."""
    t0 = np.ascontiguousarray(np.asarray(t0, np.int64))
    t1 = np.ascontiguousarray(np.asarray(t1, np.int64))
    c = np.ascontiguousarray(np.asarray(c, np.int64))
    log_ret = np.ascontiguousarray(np.asarray(log_ret, np.float64))
    w = _return_attribution_kernel(t0, t1, c, log_ret)
    tot = w.sum()
    n = len(w)
    if tot > 0:
        w = w * (n / tot)
    return w

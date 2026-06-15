"""
breaks_entropy.py, Structural-break and entropy feature kernels.

López de Prado, "Advances in Financial Machine Learning", Ch. 17 (Structural
Breaks) and Ch. 18 (Entropy Features). All features are CAUSAL: the value at
bar t uses only information from bars <= t (a backward-looking window ending at
t). No lookahead anywhere.

Contents
--------
STRUCTURAL BREAKS (Ch.17)
  * cusum_filter(y, h)        , symmetric CUSUM event sampler (Ch.2/17). Returns
                                 integer indices of the bars at which the
                                 cumulative up/down run crosses the threshold h.
  * sadf(logp...)           , Supremum Augmented Dickey-Fuller statistic on a
                                 rolling BACKWARD window. For each end-bar t we
                                 run an ADF regression for every admissible start
                                 t0 <= t - minlen and take the supremum of the
                                 t-stat on the autoregressive coefficient. This
                                 is the heavy hot loop (O(n * window * lags) OLS
                                 fits), Numba-accelerated. SADF > 0 large =>
                                 explosive / bubble regime.

ENTROPY (Ch.18)
  * quantize_signbins / quantize_qcut, causal encoders of a return string.
  * shannon_plugin(msg, w)    , plug-in (max-likelihood) Shannon entropy rate
                                 over words of length w (per-symbol, bits).
  * lempel_ziv(msg)           , LZ76 complexity (normalised) of a symbol string.
  * kontoyiannis(msg, window) , Kontoyiannis (1998) entropy-rate estimator from
                                 match lengths (the LZ/Ziv match-length loop is
                                 Numba-accelerated). bits/symbol.
  * rolling_entropy(...)      , causal rolling-window entropy feature series.

Each hot kernel has a pure-numpy/python REFERENCE and a Numba implementation;
`verify_bit_identical()` checks them against each other to machine precision.
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# Numba shim (mirror lib/bars.py): degrade gracefully if numba is missing.
# --------------------------------------------------------------------------- #
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False
    prange = range
    def njit(*a, **k):
        def deco(f):
            return f
        if a and callable(a[0]):
            return a[0]
        return deco


# =========================================================================== #
# (a) STRUCTURAL BREAKS
# =========================================================================== #
def cusum_filter(y, h):
    """Symmetric CUSUM filter event sampler (AFML Ch.2 snippet 2.4, used in Ch.17).

    Accumulates positive and negative runs of the increments of `y`; emits an
    event (and resets the relevant accumulator) whenever a run exceeds h. Causal:
    event at index i depends only on y[:i+1]. Returns int64 array of event idx.
    `h` may be a scalar or a per-bar array (e.g. k * rolling sigma)."""
    y = np.asarray(y, float)
    n = y.shape[0]
    h_arr = np.full(n, float(h)) if np.isscalar(h) else np.asarray(h, float)
    return _cusum_kernel(y, h_arr)


@njit(cache=True)
def _cusum_kernel(y, h):
    n = y.shape[0]
    out = np.empty(n, np.int64)
    m = 0
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, n):
        d = y[i] - y[i - 1]
        s_pos = s_pos + d
        if s_pos < 0.0:
            s_pos = 0.0
        s_neg = s_neg + d
        if s_neg > 0.0:
            s_neg = 0.0
        hi = h[i]
        if s_pos > hi:
            s_pos = 0.0
            out[m] = i
            m += 1
        elif s_neg < -hi:
            s_neg = 0.0
            out[m] = i
            m += 1
    return out[:m]


# --------------------------------------------------------------------------- #
# SADF, the heavy hot loop. Reference (numpy) + Numba kernels.
# --------------------------------------------------------------------------- #
def _adf_design(logp, lags):
    """Build the full ADF design once: y = Δp[t], regressors = [const, p[t-1],
    Δp[t-1..t-lags]]. Returns (yvar, X, base_index) where base_index[k] is the
    absolute bar index of row k (the end-bar t of that differenced observation).

    The ADF regression (constant, no trend) is
        Δp_t = a + b·p_{t-1} + Σ_j γ_j·Δp_{t-j} + ε_t
    and the ADF/SADF statistic is the t-stat of b. (LdP Ch.17, snippet 17.1.)"""
    logp = np.asarray(logp, float)
    dp = np.diff(logp)                      # Δp[t] aligned to absolute index t (t>=1)
    n = dp.shape[0]
    # row k corresponds to absolute bar index = lags+1+k
    rows = n - lags
    yvar = dp[lags:]                                            # Δp_t
    const = np.ones(rows)
    plag = logp[lags:lags + rows]                              # p_{t-1}
    cols = [const, plag]
    for j in range(1, lags + 1):
        cols.append(dp[lags - j:lags - j + rows])              # Δp_{t-j}
    X = np.column_stack(cols)
    base_index = np.arange(lags + 1, lags + 1 + rows)          # absolute t
    return yvar, X, base_index


def sadf_reference(logp, minlen=50, lags=1, mode="constant"):
    """Pure-numpy reference SADF on a rolling backward window.

    For each end-row r, take the supremum over all admissible start-rows s
    (window length >= minlen) of the t-stat on the AR(1) level coefficient b,
    fitting the ADF regression on rows [s, r]. Returns an array aligned to the
    ORIGINAL price index (length len(logp)); positions without a value are NaN.
    Causal: value at bar t uses only logp[:t+1]."""
    yvar, X, base_index = _adf_design(logp, lags)
    rows = yvar.shape[0]
    out_rows = np.full(rows, np.nan)
    for r in range(minlen - 1, rows):
        best = -np.inf
        for s in range(0, r - minlen + 2):
            Xs = X[s:r + 1]
            ys = yvar[s:r + 1]
            # OLS
            xtx = Xs.T @ Xs
            try:
                xtx_inv = np.linalg.inv(xtx)
            except np.linalg.LinAlgError:
                continue
            beta = xtx_inv @ (Xs.T @ ys)
            resid = ys - Xs @ beta
            dof = Xs.shape[0] - Xs.shape[1]
            if dof <= 0:
                continue
            s2 = (resid @ resid) / dof
            se_b = np.sqrt(s2 * xtx_inv[1, 1])
            if se_b > 0:
                tstat = beta[1] / se_b
                if tstat > best:
                    best = tstat
        if np.isfinite(best):
            out_rows[r] = best
    # map back to original price index
    out = np.full(logp.shape[0], np.nan)
    out[base_index] = out_rows
    return out


@njit(cache=True)
def _ols_tstat_b(X, y):
    """t-stat of the 2nd coefficient (index 1) of an OLS fit, via normal eqns."""
    p = X.shape[1]
    n = X.shape[0]
    xtx = np.zeros((p, p))
    xty = np.zeros(p)
    for i in range(n):
        for a in range(p):
            xty[a] += X[i, a] * y[i]
            for bb in range(a, p):
                xtx[a, bb] += X[i, a] * X[i, bb]
    for a in range(p):
        for bb in range(a):
            xtx[a, bb] = xtx[bb, a]
    inv = np.linalg.inv(xtx)
    beta = inv @ xty
    rss = 0.0
    for i in range(n):
        pred = 0.0
        for a in range(p):
            pred += X[i, a] * beta[a]
        e = y[i] - pred
        rss += e * e
    dof = n - p
    if dof <= 0:
        return np.nan
    s2 = rss / dof
    var_b = s2 * inv[1, 1]
    if var_b <= 0.0:
        return np.nan
    return beta[1] / np.sqrt(var_b)


@njit(cache=True, parallel=True)
def _sadf_kernel(yvar, X, minlen):
    rows = yvar.shape[0]
    out_rows = np.full(rows, np.nan)
    for r in prange(minlen - 1, rows):
        best = -np.inf
        for s in range(0, r - minlen + 2):
            t = _ols_tstat_b(X[s:r + 1], yvar[s:r + 1])
            if t == t and t > best:        # t==t rejects NaN
                best = t
        if best > -np.inf:
            out_rows[r] = best
    return out_rows


def sadf(logp, minlen=50, lags=1):
    """Numba SADF (constant-regression). Same contract as sadf_reference."""
    logp = np.asarray(logp, float)
    yvar, X, base_index = _adf_design(logp, lags)
    yvar = np.ascontiguousarray(yvar)
    X = np.ascontiguousarray(X)
    out_rows = _sadf_kernel(yvar, X, int(minlen))
    out = np.full(logp.shape[0], np.nan)
    out[base_index] = out_rows
    return out


@njit(cache=True, parallel=True)
def _sadf_capped_kernel(yvar, X, minlen, maxwin):
    """SADF with the backward window capped at `maxwin` rows: for end-row r the
    start s ranges over [max(0, r-maxwin+1), r-minlen+1]. Keeps the hot loop
    O(rows * maxwin) instead of O(rows^2). Still causal (start <= end <= r)."""
    rows = yvar.shape[0]
    out_rows = np.full(rows, np.nan)
    for r in prange(minlen - 1, rows):
        best = -np.inf
        s_lo = r - maxwin + 1
        if s_lo < 0:
            s_lo = 0
        for s in range(s_lo, r - minlen + 2):
            t = _ols_tstat_b(X[s:r + 1], yvar[s:r + 1])
            if t == t and t > best:
                best = t
        if best > -np.inf:
            out_rows[r] = best
    return out_rows


def sadf_capped_reference(logp, minlen=50, lags=1, maxwin=250):
    """Pure-numpy capped-window reference (mirror of sadf_reference + cap)."""
    yvar, X, base_index = _adf_design(logp, lags)
    rows = yvar.shape[0]
    out_rows = np.full(rows, np.nan)
    for r in range(minlen - 1, rows):
        best = -np.inf
        s_lo = max(0, r - maxwin + 1)
        for s in range(s_lo, r - minlen + 2):
            Xs = X[s:r + 1]; ys = yvar[s:r + 1]
            xtx = Xs.T @ Xs
            try:
                xtx_inv = np.linalg.inv(xtx)
            except np.linalg.LinAlgError:
                continue
            beta = xtx_inv @ (Xs.T @ ys)
            resid = ys - Xs @ beta
            dof = Xs.shape[0] - Xs.shape[1]
            if dof <= 0:
                continue
            s2 = (resid @ resid) / dof
            se_b = np.sqrt(s2 * xtx_inv[1, 1])
            if se_b > 0 and beta[1] / se_b > best:
                best = beta[1] / se_b
        if np.isfinite(best):
            out_rows[r] = best
    out = np.full(logp.shape[0], np.nan)
    out[base_index] = out_rows
    return out


# =========================================================================== #
# (b) ENTROPY FEATURES
# =========================================================================== #
def quantize_signbins(r, n_bins=2):
    """Causal symbol encoder. n_bins=2 -> binary sign string {0,1} of returns;
    n_bins=3 -> {down,flat,up} via a tiny dead-zone at 0. Each symbol at t uses
    only r[t] (already a backward return), so the message is causal."""
    r = np.asarray(r, float)
    if n_bins == 2:
        return (r > 0).astype(np.int64)
    if n_bins == 3:
        out = np.ones(r.shape[0], np.int64)             # flat
        out[r > 0] = 2
        out[r < 0] = 0
        return out
    raise ValueError("signbins supports n_bins in {2,3}")


def quantize_qcut(r, n_bins, edges):
    """Encode r into n_bins symbols using FIXED edges (no peeking). `edges` must
    be precomputed from a strictly prior window to stay causal."""
    r = np.asarray(r, float)
    return np.clip(np.searchsorted(edges, r, side="right"), 0, n_bins - 1).astype(np.int64)


def shannon_plugin(msg, word_len=1):
    """Plug-in (max-likelihood) Shannon entropy RATE in bits/symbol over words of
    length `word_len`. For word_len=1 this is the i.i.d. entropy of the symbol
    distribution; for word_len>1 it is H(block)/word_len. Reference + the same
    quantity computed by the rolling kernel. Pure-python/numpy reference."""
    msg = np.asarray(msg, np.int64)
    n = msg.shape[0]
    if n < word_len or word_len < 1:
        return np.nan
    if word_len == 1:
        counts = np.bincount(msg)
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log2(p)).sum())
    # block entropy via dictionary of windows
    from collections import Counter
    blocks = Counter()
    for i in range(n - word_len + 1):
        blocks[tuple(msg[i:i + word_len])] += 1
    tot = sum(blocks.values())
    h = 0.0
    for c in blocks.values():
        pr = c / tot
        h -= pr * np.log2(pr)
    return float(h / word_len)


def lempel_ziv_reference(msg):
    """LZ76 complexity (number of distinct phrases in the LZ parse), normalised
    by n/log2(n) so it -> entropy rate for ergodic sources. Pure-python ref
    (AFML Ch.18, snippet 18.2 style)."""
    msg = np.asarray(msg, np.int64)
    n = msg.shape[0]
    if n < 2:
        return 0.0
    s = "".join(chr(int(x)) for x in msg)              # symbols as chars
    i, c, l, k = 0, 1, 1, 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    denom = n / np.log2(n)
    return float(c / denom)


@njit(cache=True)
def _lz76_count(msg):
    """LZ76 phrase count over an integer symbol array (Numba). Mirrors
    lempel_ziv_reference's parse exactly but on the int array directly (no chr
    conversion), so it is the hot-loop-accelerated version used in the full run."""
    n = msg.shape[0]
    if n < 2:
        return 1
    i = 0
    c = 1
    l = 1
    k = 1
    k_max = 1
    while True:
        if msg[i + k - 1] == msg[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def lempel_ziv(msg):
    """Normalised LZ76 complexity via the Numba phrase-count kernel."""
    msg = np.asarray(msg, np.int64)
    n = msg.shape[0]
    if n < 2:
        return 0.0
    c = _lz76_count(msg)
    return float(c / (n / np.log2(n)))


@njit(cache=True)
def _kontoyiannis_matchlen(sym):
    """For each position i (i>=1), the length of the LONGEST prefix starting at i
    that also appears starting somewhere in sym[0:i] (the Kontoyiannis Λ_i match
    length, capped at the remaining length). Numba-accelerated string-match loop.
    Causal by construction: Λ_i looks only at the past sym[0:i]."""
    n = sym.shape[0]
    lam = np.zeros(n, np.int64)
    for i in range(1, n):
        best = 0
        for j in range(0, i):
            l = 0
            while (i + l < n) and (sym[j + l] == sym[i + l]):
                l += 1
                if j + l >= i:        # match window must stay strictly in the past start set
                    break
            if l > best:
                best = l
        lam[i] = best + 1            # Λ = matchlen + 1 (LdP/Kontoyiannis convention)
    return lam


def _kontoyiannis_matchlen_ref(sym):
    """Pure-python reference for the match-length loop above."""
    sym = np.asarray(sym, np.int64)
    n = sym.shape[0]
    lam = np.zeros(n, np.int64)
    for i in range(1, n):
        best = 0
        for j in range(0, i):
            l = 0
            while (i + l < n) and (sym[j + l] == sym[i + l]):
                l += 1
                if j + l >= i:
                    break
            if l > best:
                best = l
        lam[i] = best + 1
    return lam


def kontoyiannis(msg, use_numba=True):
    """Kontoyiannis (1998) entropy-rate estimate in bits/symbol:
        Ĥ = (1/Σ 1)·Σ_i [ log2(i) / Λ_i ]   (growing-window form, LdP Ch.18).
    Λ_i = 1 + longest-match-into-the-past. Causal."""
    msg = np.asarray(msg, np.int64)
    n = msg.shape[0]
    if n < 3:
        return np.nan
    lam = _kontoyiannis_matchlen(msg) if use_numba else _kontoyiannis_matchlen_ref(msg)
    num = 0.0
    cnt = 0
    for i in range(1, n):
        if lam[i] > 0:
            num += np.log2(i + 1) / lam[i]
            cnt += 1
    if cnt == 0:
        return np.nan
    return float(num / cnt)


# --------------------------------------------------------------------------- #
# Causal rolling-window entropy feature
# --------------------------------------------------------------------------- #
def rolling_entropy(msg, window, kind="shannon", word_len=1):
    """Causal rolling entropy: value at t uses symbols msg[t-window+1 : t+1].
    kind in {'shannon','lz','konto'}. First (window-1) entries are NaN."""
    msg = np.asarray(msg, np.int64)
    n = msg.shape[0]
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = msg[t - window + 1:t + 1]
        if kind == "shannon":
            out[t] = shannon_plugin(seg, word_len)
        elif kind == "lz":
            out[t] = lempel_ziv(seg)
        elif kind == "konto":
            out[t] = kontoyiannis(seg, use_numba=True)
        else:
            raise ValueError(kind)
    return out


# =========================================================================== #
# Verification
# =========================================================================== #
def verify_bit_identical(seed=0, n=400, verbose=True):
    """Check Numba kernels vs numpy/python references; return dict of max|Δ|.

    Covers: CUSUM event indices, SADF statistic series, and the Kontoyiannis
    match-length loop (the LZ-style hot loop)."""
    rng = np.random.default_rng(seed)
    # a series with a deliberate explosive patch so SADF has signal
    r = rng.standard_normal(n) * 0.01
    r[n // 2:n // 2 + 40] += np.linspace(0, 0.05, 40)
    logp = np.cumsum(r)

    res = {}

    # CUSUM: kernel is the only impl; sanity-check determinism + python re-derive
    h = 0.02
    idx_k = cusum_filter(logp, h)
    s_pos = s_neg = 0.0
    ref = []
    for i in range(1, n):
        d = logp[i] - logp[i - 1]
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_pos > h:
            s_pos = 0.0; ref.append(i)
        elif s_neg < -h:
            s_neg = 0.0; ref.append(i)
    res["cusum_max_abs_diff"] = float(np.max(np.abs(idx_k - np.array(ref, np.int64)))) \
        if len(ref) == len(idx_k) and len(ref) else (0.0 if len(ref) == len(idx_k) else np.inf)

    # SADF: numpy reference vs numba kernel
    a = sadf_reference(logp, minlen=40, lags=1)
    b = sadf(logp, minlen=40, lags=1)
    m = np.isfinite(a) | np.isfinite(b)
    d_sadf = np.nanmax(np.abs(a[m] - b[m]))
    res["sadf_max_abs_diff"] = float(d_sadf)

    # SADF capped-window (the kernel the full run actually uses)
    ar, br, bi = _adf_design(logp, 1)
    cap_rows_ref = sadf_capped_reference(logp, minlen=40, lags=1, maxwin=80)
    cap_nb = _sadf_capped_kernel(np.ascontiguousarray(ar), np.ascontiguousarray(br), 40, 80)
    cap_nb_full = np.full(logp.shape[0], np.nan); cap_nb_full[bi] = cap_nb
    mc = np.isfinite(cap_rows_ref) | np.isfinite(cap_nb_full)
    res["sadf_capped_max_abs_diff"] = float(np.nanmax(np.abs(cap_rows_ref[mc] - cap_nb_full[mc])))

    # Kontoyiannis match-length loop: python ref vs numba
    msg = quantize_signbins(r, 2)
    lam_ref = _kontoyiannis_matchlen_ref(msg)
    lam_nb = _kontoyiannis_matchlen(msg)
    res["konto_matchlen_max_abs_diff"] = float(np.max(np.abs(lam_ref - lam_nb)))

    # LZ76: python reference vs numba phrase-count (over rolling windows)
    diffs = []
    for t in range(120, n, 37):
        seg = msg[t - 100:t]
        diffs.append(abs(lempel_ziv_reference(seg) - lempel_ziv(seg)))
    res["lz76_max_abs_diff"] = float(max(diffs)) if diffs else 0.0

    if verbose:
        for k, v in res.items():
            print(f"  {k:32s} max|Δ| = {v:.3e}")
    return res


if __name__ == "__main__":
    print(f"numba available: {_HAVE_NUMBA}")
    print("=== verify_bit_identical (numba kernels vs references) ===")
    verify_bit_identical()

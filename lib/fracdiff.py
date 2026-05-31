"""fracdiff.py — Fixed-Width Window Fractional Differentiation (FFD).

Implements López de Prado, "Advances in Financial Machine Learning", Ch. 5.

The fractional-difference operator (1-B)^d expands, for a real exponent d, into
an infinite weighted sum of lagged values with binomial weights

    w_0 = 1,   w_k = -w_{k-1} * (d - k + 1) / k.

For 0 < d < 1 the weights decay (eventually) but never vanish; FFD truncates the
weight vector at the first k where |w_k| < tau, giving a CAUSAL fixed-width
backward-looking filter. Applied to a log-price series it yields a series that
can be stationary (passes ADF) while still correlating with the original level
(memory preserved) — the central claim of the chapter.

All transforms here are causal: output[t] uses only x[t], x[t-1], ..., x[t-W+1].
No lookahead.
"""
import numpy as np
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d, tau=1e-5, max_width=100_000):
    """Binomial FFD weights for exponent ``d``, truncated at |w_k| < tau.

    Returns a 1-D array ``w`` of length W ordered from the OLDEST lag to the
    most recent (so ``w[-1]`` multiplies x[t]); this is the convolution-ready
    orientation. Length W is the fixed window width.
    """
    w = [1.0]
    k = 1
    while k < max_width:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < tau:
            break
        w.append(w_k)
        k += 1
    # built newest-first (w[0] multiplies x[t]); reverse to oldest-first
    return np.array(w[::-1], dtype=float)


def ffd(x, d, tau=1e-5, weights=None):
    """Apply causal FFD of exponent ``d`` to 1-D array ``x``.

    Returns an array the same length as ``x`` with the first (W-1) entries NaN
    (insufficient history). ``weights`` may be passed to avoid recomputation.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = ffd_weights(d, tau)
    W = len(weights)
    out = np.full(x.shape[0], np.nan)
    if W > x.shape[0]:
        return out
    # We want a CORRELATION (no kernel flip): out[t] = sum_j weights[j]*x[t-W+1+j]
    # with weights oldest-first. np.convolve flips its 2nd arg, so feed it
    # reversed to recover the correlation.
    conv = np.convolve(x, weights[::-1], mode="valid")
    out[W - 1:] = conv
    return out


def min_d_search(x, d_grid=None, tau=1e-5, signif="5%"):
    """Find the smallest d on ``d_grid`` whose FFD series passes the ADF test.

    Returns a dict with the full per-d trace plus the selected d* and its stats.
    'pass' = ADF statistic below the chosen critical value (reject unit root).
    corr_level = Pearson corr between the FFD series and the original level x,
    computed on the overlapping (non-NaN) support — the memory measure.
    """
    x = np.asarray(x, dtype=float)
    if d_grid is None:
        d_grid = np.arange(0.0, 1.0001, 0.05)

    rows = []
    d_star = np.nan
    star_row = None
    for d in d_grid:
        w = ffd_weights(d, tau)
        W = len(w)
        y = ffd(x, d, tau=tau, weights=w)
        mask = ~np.isnan(y)
        n_obs = int(mask.sum())
        rec = dict(d=float(d), window_len=int(W), n_obs=n_obs,
                   adf_stat=np.nan, adf_p=np.nan, adf_crit_5=np.nan,
                   corr_level=np.nan, passed=False)
        if n_obs > 50:
            xv = x[mask]
            yv = y[mask]
            if np.std(yv) > 0 and np.std(xv) > 0:
                rec["corr_level"] = float(np.corrcoef(yv, xv)[0, 1])
            try:
                res = adfuller(yv, maxlag=1, regression="c", autolag=None)
                rec["adf_stat"] = float(res[0])
                rec["adf_p"] = float(res[1])
                rec["adf_crit_5"] = float(res[4][signif])
                rec["passed"] = bool(res[0] < res[4][signif])
            except Exception:
                pass
        rows.append(rec)
        if rec["passed"] and np.isnan(d_star):
            d_star = float(d)
            star_row = rec

    return dict(trace=rows, d_star=d_star, star=star_row, tau=tau,
                signif=signif, d_grid=np.asarray(d_grid))

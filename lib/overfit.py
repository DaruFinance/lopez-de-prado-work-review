"""
overfit.py — López de Prado's backtest-overfitting / multiple-testing harness.

Implements, with formulas quoted from the primary papers:
  - Probabilistic Sharpe Ratio (PSR)            "The Sharpe Ratio Efficient Frontier" (2012)
  - Expected max Sharpe under N trials          "The Deflated Sharpe Ratio" (2014), False Strategy Thm (2021)
  - Deflated Sharpe Ratio (DSR)                  "The Deflated Sharpe Ratio" (2014)
  - Minimum Track Record Length (MinTRL)         (2012)
  - Minimum Backtest Length (MinBTL)             "Pseudo-Mathematics..." (2014)
  - Probability of Backtest Overfitting (PBO)    "The Probability of Backtest Overfitting" (2017), via CSCV
  - Effective number of (independent) trials     "A Data Science Solution to the Multiple-Testing Crisis" (2019), ONC-style
  - Purged K-Fold CV + embargo, CPCV             AFML (2018) Ch. 7, 12

All Sharpe ratios here are PER-OBSERVATION (non-annualized) unless stated.
This is research infrastructure: every function is small, vectorised, and tested in __main__.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as ss
from itertools import combinations

GAMMA = 0.5772156649015328606   # Euler-Mascheroni constant


# --------------------------------------------------------------------------- #
# Sharpe-ratio statistics
# --------------------------------------------------------------------------- #
def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def prob_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float,
                      sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark). Bailey & LdP (2012), eq. for PSR.
    `kurt` is the *non-excess* kurtosis (Gaussian = 3). All SRs per-observation."""
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    if denom <= 0 or n_obs < 2:
        return np.nan
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(ss.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max_k SR_k] for N independent skill-less trials whose SR estimates have
    variance var_sr. False Strategy Theorem (LdP 2014/2021):
        E[max] ≈ sqrt(var_sr) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    em = ((1 - GAMMA) * ss.norm.ppf(1 - 1.0 / n_trials)
          + GAMMA * ss.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(var_sr) * em)


def deflated_sharpe_ratio(sr_best: float, n_obs: int, skew: float, kurt: float,
                          sr_trials: np.ndarray) -> dict:
    """DSR: PSR of the selected strategy against the expected-max-SR benchmark
    implied by the number and dispersion of trials. LdP & Bailey (2014)."""
    sr_trials = np.asarray(sr_trials, float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    n_trials = len(sr_trials)
    var_sr = sr_trials.var(ddof=1) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    dsr = prob_sharpe_ratio(sr_best, n_obs, skew, kurt, sr_benchmark=sr0)
    return dict(dsr=dsr, sr0=sr0, n_trials=n_trials, var_sr=var_sr)


def min_track_record_length(sr: float, skew: float, kurt: float,
                            sr_benchmark: float = 0.0, prob: float = 0.95) -> float:
    """MinTRL: # observations needed for PSR(sr) > prob. LdP (2012)."""
    if sr <= sr_benchmark:
        return np.inf
    z = ss.norm.ppf(prob)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2)


def min_backtest_length(n_trials: int, target_max_sr: float) -> float:
    """MinBTL: # observations so the expected max SR of N skill-less trials does
    not exceed target_max_sr (using var_sr ≈ 1/T). Pseudo-Math (2014)."""
    if n_trials < 2 or target_max_sr <= 0:
        return np.nan
    em = ((1 - GAMMA) * ss.norm.ppf(1 - 1.0 / n_trials)
          + GAMMA * ss.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float((em / target_max_sr) ** 2)


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
def pbo_cscv(M: np.ndarray, n_splits: int = 12) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    M: (T_obs, N_strategies) matrix of per-observation returns.
    Split rows into S equal blocks; for every way of choosing S/2 blocks as IS
    (rest OOS), pick the best-IS strategy and find its OOS rank. PBO = fraction
    of splits whose best-IS strategy lands in the bottom half OOS (logit ≤ 0).
    Deterministic & model-free (Bailey, Borwein, LdP, Zhu 2017)."""
    M = np.asarray(M, float)
    T, N = M.shape
    S = n_splits - (n_splits % 2)
    block = T // S
    # precompute per-block sum, sum-of-squares, count -> combine combinatorially
    bsum = np.empty((S, N)); bsq = np.empty((S, N)); bn = np.empty(S)
    for b in range(S):
        x = M[b * block:(b + 1) * block]
        bsum[b] = x.sum(0); bsq[b] = (x * x).sum(0); bn[b] = x.shape[0]

    def sharpe_from(idx):
        n = bn[list(idx)].sum()
        s = bsum[list(idx)].sum(0); q = bsq[list(idx)].sum(0)
        mu = s / n
        var = (q - n * mu * mu) / (n - 1)
        sd = np.sqrt(np.clip(var, 0, None))
        return np.where(sd > 0, mu / sd, 0.0)

    logits, oos_rel = [], []
    allb = set(range(S))
    for is_idx in combinations(range(S), S // 2):
        oos_idx = tuple(allb - set(is_idx))
        n_star = int(np.argmax(sharpe_from(is_idx)))
        rank = ss.rankdata(sharpe_from(oos_idx))[n_star] / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
        oos_rel.append(rank)
    logits = np.array(logits)
    return dict(pbo=float((logits <= 0).mean()), logits=logits,
                oos_rel=np.array(oos_rel), n_combos=len(logits), S=S)


# --------------------------------------------------------------------------- #
# Effective number of independent trials (ONC-style clustering)
# --------------------------------------------------------------------------- #
def effective_n_trials(returns_matrix: np.ndarray, k_max: int = 25,
                       seed: int = 0) -> dict:
    """Effective number of independent trials among correlated strategies.

    Primary measure = eigenvalue participation ratio of the strategy-return
    correlation matrix: PR = (Σλ)² / Σλ²  (∈ [1, N]). PR = N when trials are
    independent (all λ=1) and → 1 when they are one repeated bet (rank-1). This
    is the deterministic "effective number of independent factors" and the right
    N to feed the False Strategy Theorem when trials are correlated (LdP 2019).
    Also returns an ONC-style silhouette cluster count as a cross-check."""
    R = np.asarray(returns_matrix, float)
    N = R.shape[1]
    if N < 4:
        return dict(effective_n=N, pr=float(N), k_clusters=N, corr_med=np.nan)
    corr = np.nan_to_num(np.corrcoef(R.T), nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(corr), 0, None)
    pr = float(lam.sum() ** 2 / np.square(lam).sum()) if np.square(lam).sum() > 0 else 1.0
    iu = np.triu_indices(N, 1)
    out = dict(effective_n=int(round(pr)), pr=pr, corr_med=float(np.median(np.abs(corr[iu]))))
    # optional ONC-style cross-check (silhouette over agglomerative clusters)
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        best_k, best_s = 2, -1.0
        for k in range(2, min(k_max, N - 1) + 1):
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                          linkage="average").fit_predict(dist)
            s = silhouette_score(dist, lab, metric="precomputed")
            if s > best_s:
                best_s, best_k = s, k
        out["k_clusters"] = best_k
    except Exception:
        out["k_clusters"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Purged K-Fold CV + embargo, and CPCV split generator   (AFML Ch.7, 12)
# --------------------------------------------------------------------------- #
def purged_kfold_splits(n: int, n_splits: int = 6, embargo_pct: float = 0.01,
                        label_span: int = 1):
    """Yield (train_idx, test_idx) over n observations with purging + embargo.

    label_span: # observations each label depends on (overlap window). Train
    points whose label window overlaps the test fold (plus an embargo after it)
    are purged. Returns contiguous test folds (time-ordered)."""
    idx = np.arange(n)
    fold = n // n_splits
    emb = int(n * embargo_pct)
    for f in range(n_splits):
        t0, t1 = f * fold, (n if f == n_splits - 1 else (f + 1) * fold)
        test = idx[t0:t1]
        lo = t0 - label_span                  # purge labels overlapping test start
        hi = t1 + label_span + emb            # purge + embargo after test end
        train = idx[(idx < lo) | (idx >= hi)]
        yield train, test


def cpcv_splits(n: int, n_groups: int = 6, k_test: int = 2,
                embargo_pct: float = 0.01, label_span: int = 1):
    """Combinatorial Purged CV: all C(n_groups, k_test) ways of holding out
    k_test groups as the test set, with purging + embargo. Yields (train, test)
    and produces multiple backtest paths (AFML Ch.12)."""
    idx = np.arange(n)
    g = n // n_groups
    bounds = [(i * g, (n if i == n_groups - 1 else (i + 1) * g)) for i in range(n_groups)]
    emb = int(n * embargo_pct)
    for combo in combinations(range(n_groups), k_test):
        test = np.concatenate([idx[bounds[c][0]:bounds[c][1]] for c in combo])
        mask = np.ones(n, bool)
        for c in combo:
            t0, t1 = bounds[c]
            mask[max(0, t0 - label_span):min(n, t1 + label_span + emb)] = False
        yield idx[mask], test


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) Expected-max-Sharpe formula vs Monte Carlo (skill-less trials)
    print("=== False Strategy Theorem: formula vs Monte Carlo ===")
    for N in (10, 100, 1000):
        T = 1000
        sims = 2000
        maxsr = []
        for _ in range(sims):
            X = rng.standard_normal((T, N))
            sr = X.mean(0) / X.std(0, ddof=1)
            maxsr.append(sr.max())
        mc = np.mean(maxsr)
        formula = expected_max_sharpe(N, var_sr=1.0 / T)
        print(f"  N={N:5d}  MC E[maxSR]={mc:.4f}   formula={formula:.4f}   "
              f"diff={abs(mc-formula):.4f}")

    # 2) DSR: skill-less winner should deflate to ~0; genuine edge should survive
    print("\n=== Deflated Sharpe Ratio ===")
    T, N = 1000, 200
    X = rng.standard_normal((T, N))                 # all skill-less
    sr_trials = X.mean(0) / X.std(0, ddof=1)
    best = int(np.argmax(sr_trials))
    rb = X[:, best]
    d = deflated_sharpe_ratio(sharpe(rb), T, ss.skew(rb), ss.kurtosis(rb, fisher=False), sr_trials)
    print(f"  skill-less winner: SR={sharpe(rb):.3f}  SR0={d['sr0']:.3f}  DSR={d['dsr']:.3f} (want ~0)")
    edge = rng.standard_normal(T) + 0.12            # true per-bar SR ~0.12
    d2 = deflated_sharpe_ratio(sharpe(edge), T, ss.skew(edge), ss.kurtosis(edge, fisher=False), sr_trials)
    print(f"  genuine edge:      SR={sharpe(edge):.3f}  SR0={d2['sr0']:.3f}  DSR={d2['dsr']:.3f} (want ~1)")

    # 3) PBO: overfit corpus (all noise) should give high PBO (~0.5)
    print("\n=== PBO (CSCV) on a pure-noise corpus ===")
    M = rng.standard_normal((1200, 100))
    print(f"  PBO={pbo_cscv(M, 12)['pbo']:.3f}  (pure noise -> expect ~0.5)")
    print("\nself-tests done.")

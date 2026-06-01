"""
barstats.py, statistical properties LdP uses to judge a bar type (AFML Ch.2).

His claims, made testable:
  (1) Information-driven bars (esp. dollar bars) produce returns closer to IID
      Gaussian than time bars -> lower |skew|, lower excess kurtosis, higher
      Jarque-Bera normality, weaker serial correlation.
  (2) Dollar-bar *counts* per calendar period are more stable through time than
      tick or volume bar counts (robust to price drift / activity regimes).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as ss
from statsmodels.stats.diagnostic import acorr_ljungbox


def return_stats(r: pd.Series) -> dict:
    """Normality / IID diagnostics for a return series."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 50:
        return dict(n=n, skew=np.nan, exkurt=np.nan, jb=np.nan, jb_p=np.nan,
                    ac1=np.nan, lb_p=np.nan)
    z = (r - r.mean()) / r.std(ddof=1)
    jb, jb_p = ss.jarque_bera(r.to_numpy())
    ac1 = pd.Series(r).autocorr(lag=1)
    lb = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    return dict(
        n=int(n),
        skew=float(ss.skew(r)),
        exkurt=float(ss.kurtosis(r, fisher=True)),   # excess kurtosis (Gaussian=0)
        jb=float(jb), jb_p=float(jb_p),
        ac1=float(ac1), lb_p=float(lb_p),
        std_returns=z,                               # for QQ / histogram plots
    )


def count_stability(bars: pd.DataFrame, freq: str = "W") -> dict:
    """Coefficient of variation of bars-per-period (lower = more stable)."""
    per = bars.groupby(pd.Grouper(freq=freq)).size()
    per = per[per > 0]
    if len(per) < 5:
        return dict(cv=np.nan, mean=np.nan, periods=len(per), series=per)
    cv = float(per.std(ddof=1) / per.mean())
    return dict(cv=cv, mean=float(per.mean()), periods=int(len(per)), series=per)

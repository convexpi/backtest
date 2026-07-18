"""Resampling for dependent data.

Ordinary bootstrap assumes i.i.d. rows; financial returns are serially dependent, so we
use the **stationary block bootstrap** (Politis & Romano, 1994): resample blocks of random
geometric length so the resample preserves short-range dependence. Use it for confidence
intervals on any metric, and `newey_west_tstat` for autocorrelation-robust IC significance."""
from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(n: int, avg_block: int = 20, rng=None) -> np.ndarray:
    """Indices for one stationary-bootstrap resample of length n (geometric block lengths,
    circular wrap). `avg_block` is the mean block length."""
    rng = rng or np.random.default_rng()
    p = 1.0 / max(1, avg_block)
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = rng.geometric(p)
        for j in range(length):
            if i >= n:
                break
            idx[i] = (start + j) % n
            i += 1
    return idx


def bootstrap_ci(returns, statistic, n_boot: int = 1000, avg_block: int = 20,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Block-bootstrap confidence interval for `statistic(returns)` (e.g. `sharpe`)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    rng = np.random.default_rng(seed)
    boot = np.array([statistic(r[stationary_bootstrap_indices(n, avg_block, rng)])
                     for _ in range(n_boot)], dtype=float)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"point": float(statistic(r)), "lo": float(lo), "hi": float(hi),
            "std": float(boot.std(ddof=1))}


def newey_west_tstat(x, y=None, lags: int | None = None) -> float:
    """Autocorrelation-robust (Newey-West) t-stat that the mean of `x` (or the covariance
    of `x` and `y`) differs from zero — for judging the significance of an IC series whose
    period-by-period values are serially correlated."""
    a = np.asarray(x, dtype=float)
    series = a if y is None else (a - a.mean()) * (np.asarray(y, float) - np.asarray(y, float).mean())
    series = series[np.isfinite(series)]
    n = series.size
    if n < 3:
        return float("nan")
    mu = series.mean()
    e = series - mu
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        cov = (e[L:] @ e[:-L]) / n
        var += 2 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se)

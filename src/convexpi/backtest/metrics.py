"""Evaluation metrics with the multiple-testing correction built in.

The distinctive piece here is the *deflated* Sharpe ratio (Bailey & Lopez de Prado,
2014): a Sharpe is not evidence until it is discounted for how many strategies were
tried to find it and for the shortness and non-normality of the sample. NumPy only —
the two normal-distribution helpers are inlined so the package has no heavy deps.
"""
from __future__ import annotations

import math

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe(returns, periods: int = 252) -> float:
    """Annualized Sharpe ratio of a per-period return series (excess assumed)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline of the cumulative return path (a negative number)."""
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def rank_ic(predictions, realized) -> float:
    """Spearman rank correlation between predictions and realized outcomes."""
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(realized, dtype=float)
    mask = ~(np.isnan(p) | np.isnan(a))
    p, a = p[mask], a[mask]
    if p.size < 2:
        return float("nan")
    pr = np.argsort(np.argsort(p)).astype(float)
    ar = np.argsort(np.argsort(a)).astype(float)
    pr -= pr.mean(); ar -= ar.mean()
    denom = np.sqrt((pr**2).sum() * (ar**2).sum())
    return float((pr * ar).sum() / denom) if denom else float("nan")


def turnover(weights) -> float:
    """Average one-period gross turnover of a (T, N) weight matrix."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean())


def net_returns(gross_returns, weights, cost_bps: float = 10.0):
    """Charge `cost_bps` per unit of turnover against a gross return series."""
    g = np.asarray(gross_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    tvr = np.concatenate([[0.0], np.abs(np.diff(w, axis=0)).sum(axis=1)])
    return g - (cost_bps / 1e4) * tvr


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true Sharpe > benchmark) given a per-period SR estimate (Bailey & LdP).

    `sr` and `sr_benchmark` are per-observation Sharpe ratios (not annualized).
    """
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe under the null of no skill, from n_trials
    independent tries (the benchmark the deflated Sharpe must clear)."""
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    g = EULER_MASCHERONI
    return sr_trials_std * ((1 - g) * _norm_ppf(1 - 1.0 / n_trials)
                            + g * _norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(sr: float, sr_trials_std: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe ratio: PSR against the expected-max-Sharpe benchmark implied by
    the number of trials. `sr` is per-observation. Below ~0.95 the "edge" is not credible
    once the search is accounted for."""
    sr0 = expected_max_sharpe(sr_trials_std, n_trials)
    return probabilistic_sharpe(sr, sr0, n_obs, skew, kurt)

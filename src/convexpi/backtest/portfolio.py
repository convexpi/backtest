"""Turning a signal into positions, and the risk of the result.

Sizing (`vol_target`, `risk_parity`, `kelly_fraction`), constraints (leverage / position
caps), and downside risk (`value_at_risk`, `expected_shortfall`, `max_drawdown`). Naive
mean-variance is deliberately *not* the default — it error-maximizes; robust sizing is."""
from __future__ import annotations

import numpy as np

from .metrics import max_drawdown  # re-exported for convenience


def vol_target(returns, target_vol: float = 0.10, window: int = 60, periods: int = 252):
    """Scale a strategy's returns to a constant annualized volatility using a trailing,
    point-in-time volatility estimate (no look-ahead — the scale for day t uses data < t)."""
    r = np.asarray(returns, dtype=float)
    scaled = np.zeros_like(r)
    for t in range(len(r)):
        lo = max(0, t - window)
        vol = r[lo:t].std(ddof=1) * np.sqrt(periods) if t - lo >= 2 else np.nan
        scaled[t] = r[t] * (target_vol / vol) if vol and np.isfinite(vol) and vol > 0 else 0.0
    return scaled


def risk_parity(cov: np.ndarray, iters: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Long-only equal-risk-contribution weights for a covariance matrix (iterative)."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (cov @ w)                     # risk contributions
        target = (w @ cov @ w) / n
        grad = rc - target
        w = np.clip(w - 0.01 * grad / (np.diag(cov) + 1e-12), 1e-6, None)
        w /= w.sum()
        if np.abs(grad).max() < tol:
            break
    return w


def kelly_fraction(mean: float, variance: float, cap: float = 1.0) -> float:
    """Kelly-optimal fraction f* = mean / variance, capped (full Kelly is famously too
    aggressive on estimated moments — cap at a fraction)."""
    if variance <= 0:
        return 0.0
    return float(np.clip(mean / variance, -cap, cap))


def apply_constraints(weights, max_leverage: float = 1.0, max_position: float = 1.0):
    """Cap per-name weights and rescale to a gross-leverage budget."""
    w = np.asarray(weights, dtype=float).copy()
    w = np.clip(w, -max_position, max_position)
    gross = np.abs(w).sum()
    if gross > max_leverage and gross > 0:
        w *= max_leverage / gross
    return w


def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk (a negative number): the alpha-quantile loss."""
    return float(np.quantile(np.asarray(returns, dtype=float), alpha))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """Expected shortfall / CVaR: mean loss in the worst alpha tail."""
    r = np.asarray(returns, dtype=float)
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else float(var)

"""Transaction costs and capacity — a prediction is not a profit.

A gross Sharpe is a hypothesis; the net Sharpe is the claim. `CostModel` charges a half-
spread, a commission, and a **square-root market-impact** term (impact grows with the
square root of participation), and `capacity` estimates the AUM at which impact eats the
edge. (Rule 14; the "trading is not free" force of Chapter 1.)"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """Per-trade cost in return units. `spread_bps` is the full bid-ask spread (half is
    paid), `commission_bps` a linear fee, and `impact_coef` scales the square-root impact
    on participation rate (fraction of ADV traded)."""
    spread_bps: float = 5.0
    commission_bps: float = 1.0
    impact_coef: float = 10.0          # bps at 100% of ADV

    def cost(self, turnover, participation=0.0):
        """Cost (in return units) for a given per-period turnover and participation rate."""
        turnover = np.asarray(turnover, dtype=float)
        part = np.asarray(participation, dtype=float)
        linear = (0.5 * self.spread_bps + self.commission_bps) / 1e4
        impact = self.impact_coef / 1e4 * np.sqrt(np.clip(part, 0, None))
        return turnover * (linear + impact)

    def net(self, gross_returns, weights, adv_fraction=0.0):
        """Charge costs against a gross return series given a (T[,N]) weight path."""
        g = np.asarray(gross_returns, dtype=float)
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        tvr = np.concatenate([[0.0], np.abs(np.diff(w, axis=0)).sum(axis=1)])
        return g - self.cost(tvr, adv_fraction)


def capacity(gross_sharpe: float, avg_turnover: float, adv_usd: float,
             cost_model: CostModel | None = None, periods: int = 252) -> float:
    """Rough AUM (USD) at which square-root impact halves the gross Sharpe.

    Solves for the participation rate whose impact cost per period equals half the gross
    per-period edge, then maps participation to AUM via average daily volume.
    """
    cm = cost_model or CostModel()
    edge_per_period = gross_sharpe / np.sqrt(periods)          # per-period SR ~ mean/vol
    if edge_per_period <= 0 or avg_turnover <= 0:
        return 0.0
    # impact_bps(part) * turnover = half the edge (in bps of vol ~ treat edge in bps)
    target_impact = 0.5 * edge_per_period * 1e4 / max(avg_turnover, 1e-9)
    part = (target_impact / cm.impact_coef) ** 2               # invert sqrt law
    return float(np.clip(part, 0, 1) * adv_usd)

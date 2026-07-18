"""Count everything you try, so the multiple-testing math is honest.

The number of configurations searched is an input to the deflated Sharpe ratio, and it
is the one input people fudge. `TrialRegistry` makes it a byproduct of running trials:
you cannot forget to count what the object counted for you (Rule 14)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics


@dataclass
class Trial:
    name: str
    sharpe: float
    reason: str
    meta: dict = field(default_factory=dict)


class TrialRegistry:
    """Append-only log of every strategy/config evaluated in a search.

    >>> reg = TrialRegistry(periods=252)
    >>> for cfg in grid: reg.log(cfg.name, returns=bt(cfg), reason="grid over lookback")
    >>> reg.deflated_best(n_obs=len(returns))   # PSR of the winner, deflated for len(grid)
    """

    def __init__(self, periods: int = 252):
        self.periods = periods
        self.trials: list[Trial] = []

    def log(self, name: str, returns=None, sharpe: float | None = None,
            reason: str = "", **meta) -> Trial:
        """Record one trial by its return series (preferred) or a precomputed Sharpe."""
        if not reason:
            raise ValueError(f"trial {name!r}: say why you tried it (reason=...)")
        if sharpe is None:
            if returns is None:
                raise ValueError("pass returns= or sharpe=")
            sharpe = metrics.sharpe(returns, self.periods)
        t = Trial(name=name, sharpe=float(sharpe), reason=reason, meta=meta)
        self.trials.append(t)
        return t

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def best(self) -> Trial:
        return max(self.trials, key=lambda t: (t.sharpe if np.isfinite(t.sharpe) else -np.inf))

    def sharpe_std(self) -> float:
        """Cross-trial dispersion of the (annualized) Sharpe estimates."""
        s = np.array([t.sharpe for t in self.trials], dtype=float)
        s = s[np.isfinite(s)]
        return float(s.std(ddof=1)) if s.size >= 2 else 0.0

    def deflated_best(self, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
        """Deflated Sharpe of the best trial, using the logged trial count and dispersion.

        Returns a probability in [0, 1]; below ~0.95 the winner is not credible once the
        search is accounted for. Sharpe is de-annualized to per-observation for the PSR.
        """
        best = self.best()
        sr_obs = best.sharpe / np.sqrt(self.periods)
        std_obs = self.sharpe_std() / np.sqrt(self.periods)
        return metrics.deflated_sharpe(sr_obs, std_obs, self.n_trials, n_obs, skew, kurt)

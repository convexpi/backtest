"""Overfitting statistics: how much of a backtest is selection, not skill.

- **PBO** (probability of backtest overfitting) via combinatorially-symmetric CV: across
  many in-sample/out-of-sample splits, how often does the in-sample-best strategy land
  below the OOS median? (Bailey, Borwein, Lopez de Prado & Zhu, 2017.)
- **Haircut Sharpe** (Harvey & Liu, 2015): a multiple-testing haircut on the Sharpe.
- **Minimum backtest length** and **White's Reality Check** (bootstrap) round it out.

The deflated Sharpe ratio itself lives in `metrics`."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .metrics import sharpe, _norm_cdf


def pbo(perf: np.ndarray, n_splits: int = 16, periods: int = 252) -> dict:
    """Probability of backtest overfitting from a (T, N) matrix of N strategies' per-period
    returns. Ranks strategies by Sharpe in each in-sample half and checks the OOS rank of
    the in-sample winner. Returns {'pbo', 'n_combos'}; PBO near 0 is good, near 1 is a
    selection artifact. N should be >= 2 strategies, n_splits even."""
    perf = np.asarray(perf, dtype=float)
    T, N = perf.shape
    if N < 2:
        raise ValueError("need >= 2 candidate strategies")
    S = n_splits - (n_splits % 2)
    blocks = np.array_split(np.arange(T), S)

    def block_sharpe(idx):
        return np.array([sharpe(perf[idx, j], periods) for j in range(N)])

    logits = []
    for is_combo in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in is_combo])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in is_combo])
        is_s, oos_s = block_sharpe(is_idx), block_sharpe(oos_idx)
        n_star = int(np.nanargmax(is_s))
        rank = (np.argsort(np.argsort(oos_s))[n_star] + 1) / (N + 1)  # relative OOS rank
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": len(logits)}


def haircut_sharpe(sr: float, n_trials: int, n_obs: int, periods: int = 252) -> float:
    """Bonferroni-style multiple-testing haircut on an annualized Sharpe (Harvey & Liu).

    Converts the Sharpe to a t-stat, adjusts its p-value for `n_trials`, and maps back —
    the fraction of the Sharpe that survives. Returns the haircut annualized Sharpe.
    """
    if sr <= 0 or n_obs < 2:
        return max(sr, 0.0)
    t = sr / np.sqrt(periods) * np.sqrt(n_obs)          # per-obs SR -> t-stat
    p_single = 1 - _norm_cdf(t)
    p_multi = min(1.0, p_single * n_trials)             # Bonferroni
    if p_multi >= 0.5:
        return 0.0
    from .metrics import _norm_ppf
    t_adj = _norm_ppf(1 - p_multi)
    return float(sr * (t_adj / t))


def min_backtest_length(sharpe_annual: float, periods: int = 252) -> float:
    """Minimum track-record length (in years) for an annualized Sharpe to be distinguishable
    from zero at ~95% — a quick reality check on short, impressive backtests."""
    if sharpe_annual <= 0:
        return float("inf")
    # need t = SR*sqrt(T_years) >= ~2  =>  T_years >= (2/SR)^2  (per-year SR)
    return float((2.0 / sharpe_annual) ** 2)


def reality_check(strategy_returns, benchmark_returns=None, n_boot: int = 1000,
                  seed: int = 0, periods: int = 252) -> float:
    """White's Reality Check p-value: is the *best* of several strategies better than the
    benchmark after data snooping? Pass a (T, N) matrix. Stationary-bootstrap the mean
    excess performance and compare the observed max to the bootstrap null."""
    X = np.asarray(strategy_returns, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, N = X.shape
    bench = np.zeros(T) if benchmark_returns is None else np.asarray(benchmark_returns, float)
    excess = X - bench.reshape(-1, 1)
    f_bar = excess.mean(0)
    v_obs = np.sqrt(T) * f_bar.max()
    rng = np.random.default_rng(seed)
    from .bootstrap import stationary_bootstrap_indices
    null = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, rng=rng)
        resampled = excess[idx].mean(0) - f_bar          # center under the null
        null[b] = np.sqrt(T) * resampled.max()
    return float((null >= v_obs).mean())

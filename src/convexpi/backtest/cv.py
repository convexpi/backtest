"""Combinatorial purged cross-validation (Lopez de Prado).

Walk-forward gives one out-of-sample path. **Combinatorial purged CV** (CPCV) gives a
whole *distribution* of them: split the series into G groups, use every combination of k
groups as the test set (purging and embargoing training rows whose labels overlap), and
reassemble the held-out predictions into many backtest paths. A strategy whose edge
depends on *which* path you happened to run is overfit."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


def _blocked(n, test_idx, groups, test_combo, embargo, label_horizon):
    """Boolean mask of training rows to drop: label-window overlap (purge) + embargo."""
    drop = np.zeros(n, dtype=bool)
    drop[test_idx] = True
    for g in test_combo:
        t0, t1 = groups[g][0], groups[g][-1]
        drop[max(0, t0 - label_horizon): t1 + 1] = True          # purge overlapping labels
        drop[t1 + 1: min(n, t1 + 1 + embargo)] = True             # embargo after the block
    return drop


def purged_kfold(n: int, n_splits: int = 5, embargo: int = 0, label_horizon: int = 1):
    """K contiguous test folds; training is everything else, purged + embargoed."""
    groups = np.array_split(np.arange(n), n_splits)
    for k in range(n_splits):
        test_idx = groups[k]
        drop = _blocked(n, test_idx, groups, (k,), embargo, label_horizon)
        train_idx = np.arange(n)[~drop]
        yield train_idx, np.asarray(test_idx)


def combinatorial_purged_split(n: int, n_groups: int = 6, n_test_groups: int = 2,
                               embargo: int = 0, label_horizon: int = 1):
    """Yield (train_idx, test_idx) for every choice of `n_test_groups` of `n_groups`.

    There are C(n_groups, n_test_groups) splits. Training rows overlapping any test block
    (within `label_horizon`) or inside the `embargo` after it are dropped.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("need 1 <= n_test_groups < n_groups")
    groups = np.array_split(np.arange(n), n_groups)
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_combo])
        drop = _blocked(n, test_idx, groups, test_combo, embargo, label_horizon)
        train_idx = np.arange(n)[~drop]
        yield train_idx, np.sort(test_idx)


def n_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct backtest paths CPCV reconstructs: k * C(G, k) / G."""
    return n_test_groups * comb(n_groups, n_test_groups) // n_groups


def cpcv_paths(n: int, predict, n_groups: int = 6, n_test_groups: int = 2,
               embargo: int = 0, label_horizon: int = 1) -> np.ndarray:
    """Run `predict(train_idx, test_idx) -> array over test_idx` across all CPCV splits and
    assemble the per-group predictions into the full set of backtest paths.

    Returns an array of shape (n_paths, n) where each row is one reconstructed OOS path
    (NaN where that group was not covered by this path). Feed each row to your metrics to
    see the *dispersion* of out-of-sample performance, not a single number.
    """
    groups = np.array_split(np.arange(n), n_groups)
    combos = list(combinations(range(n_groups), n_test_groups))
    # per (combo, group) predictions
    preds = {}
    for test_combo in combos:
        test_idx = np.concatenate([groups[g] for g in test_combo])
        drop = _blocked(n, test_idx, groups, test_combo, embargo, label_horizon)
        train_idx = np.arange(n)[~drop]
        out = np.asarray(predict(train_idx, np.sort(test_idx)), dtype=float)
        pos = 0
        for g in test_combo:
            preds[(test_combo, g)] = out[pos: pos + len(groups[g])]
            pos += len(groups[g])
    # each group is tested in C(G-1, k-1) combos; distribute into that many paths
    n_paths = n_backtest_paths(n_groups, n_test_groups)
    paths = np.full((n_paths, n), np.nan)
    group_fill = {g: 0 for g in range(n_groups)}
    for test_combo in combos:
        for g in test_combo:
            p = group_fill[g]
            paths[p, groups[g]] = preds[(test_combo, g)]
            group_fill[g] += 1
    return paths

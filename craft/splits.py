"""Time-aware splitters. Walk-forward is the default; random k-fold is not offered,
because on ordered, serially dependent data it leaks the future (Rule 3). When labels
span multiple periods, `purged_walk_forward` drops training rows adjacent to the test
block (purge) and holds out an embargo gap after it (Lopez de Prado)."""
from __future__ import annotations

import numpy as np


def walk_forward(n: int, n_splits: int = 5, min_train: int | None = None,
                 rolling: bool = False):
    """Yield (train_idx, test_idx) forward in time.

    The series is cut into `n_splits` contiguous test blocks; training is everything
    strictly before each block (expanding), or a window of the same length (rolling).
    """
    if n_splits < 1 or n_splits >= n:
        raise ValueError("need 1 <= n_splits < n")
    fold = n // (n_splits + 1)
    if min_train is None:
        min_train = fold
    for k in range(1, n_splits + 1):
        test_start = min_train + (k - 1) * fold
        test_end = min(test_start + fold, n)
        if test_start >= n:
            break
        train_start = max(0, test_start - min_train) if rolling else 0
        train_idx = np.arange(train_start, test_start)
        test_idx = np.arange(test_start, test_end)
        if train_idx.size and test_idx.size:
            yield train_idx, test_idx


def purged_walk_forward(n: int, n_splits: int = 5, embargo: int = 0,
                        label_horizon: int = 1, min_train: int | None = None):
    """Walk-forward with purge + embargo for overlapping labels.

    `label_horizon` is how many periods a label spans; training rows whose label window
    overlaps the test block are purged, and `embargo` further rows after the test block
    are withheld from any later training fold.
    """
    for train_idx, test_idx in walk_forward(n, n_splits, min_train):
        t0, t1 = test_idx[0], test_idx[-1]
        purge_lo = t0 - label_horizon
        keep = train_idx[(train_idx < purge_lo) | (train_idx > t1 + embargo)]
        yield keep, test_idx

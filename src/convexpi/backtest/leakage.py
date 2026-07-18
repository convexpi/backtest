"""Automated leakage detectors — catch the future before it flatters you.

- `shuffle_target_test`: permute the labels and refit; if the real score is not clearly
  above the permuted null, there is no signal (or the "signal" is leakage the permutation
  can't remove).
- `time_shuffle_test`: if shuffling the row order *improves* cross-validated score, the
  pipeline is exploiting temporal order — leakage.
- `target_leakage_scan`: features suspiciously correlated with the target.
- `training_window_gradient`: does OOS skill grow with more history (learning) or peak at
  the shortest window (recency)? A discriminant between the two."""
from __future__ import annotations

import numpy as np


def shuffle_target_test(score_fn, y, n_permutations: int = 100, seed: int = 0) -> dict:
    """Permutation test. `score_fn(y)` returns an OOS score for labels y. Returns the real
    score, the permuted-null mean, and a p-value = P(null >= real)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    real = float(score_fn(y))
    null = np.array([float(score_fn(rng.permutation(y))) for _ in range(n_permutations)])
    return {"score": real, "null_mean": float(null.mean()),
            "p_value": float((null >= real).mean())}


def time_shuffle_test(ordered_score: float, shuffled_score: float,
                      tol: float = 0.02) -> dict:
    """Flag temporal leakage: if a shuffled-CV score materially exceeds the time-ordered CV
    score, the model is peeking across time. Pass the two scores you already computed."""
    leak = shuffled_score - ordered_score > tol
    return {"ordered": ordered_score, "shuffled": shuffled_score,
            "leak_suspected": bool(leak),
            "detail": "shuffled CV beats time-ordered CV — likely look-ahead"
                      if leak else "ok: shuffling does not help"}


def target_leakage_scan(X, y, threshold: float = 0.95) -> list[tuple[int, float]]:
    """Return (column, |correlation|) for features whose absolute correlation with the
    target exceeds `threshold` — a near-perfect correlate is usually the label in disguise."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    flags = []
    for j in range(X.shape[1]):
        c = np.corrcoef(X[:, j], y)[0, 1]
        if np.isfinite(c) and abs(c) >= threshold:
            flags.append((j, float(abs(c))))
    return flags


def training_window_gradient(fit_eval, windows) -> dict:
    """`fit_eval(window)` returns an OOS score for a model trained on the last `window`
    observations. A positive slope of score vs window is the fingerprint of *learning*
    (more history helps); a negative slope is *recency* (short windows win)."""
    w = np.asarray(list(windows), dtype=float)
    s = np.array([float(fit_eval(int(win))) for win in w], dtype=float)
    slope = float(np.polyfit(w, s, 1)[0]) if np.isfinite(s).all() else float("nan")
    verdict = "learning" if slope > 0 else "recency"
    return {"windows": w.tolist(), "scores": s.tolist(), "slope": slope, "verdict": verdict}

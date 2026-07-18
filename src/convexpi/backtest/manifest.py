"""A reproducibility manifest to stamp on every result.

Seed, a content hash of the input data, package versions, and the git commit — the
minimum needed for someone else (or you, in six months) to rerun the backtest and get the
same number. This is the "environment file" discipline, made one function call."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np


def data_hash(obj) -> str:
    """A stable content hash of an array or DataFrame-like object (first 16 hex chars)."""
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            obj = obj.to_numpy()
    except Exception:
        pass
    arr = np.ascontiguousarray(np.asarray(obj))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    vers = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            vers[name] = __import__(name).__version__
        except Exception:
            pass
    return vers


def manifest(seed=None, data=None, when: str | None = None, **extra) -> dict:
    """Build a reproducibility record. Pass the seed you used and the input data.

    `when` lets callers inject a timestamp for deterministic tests; otherwise it is
    stamped in UTC now.
    """
    m = {
        "seed": seed,
        "data_hash": data_hash(data) if data is not None else None,
        "git_sha": _git_sha(),
        "versions": _versions(),
        "timestamp": when or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    m.update(extra)
    return m

"""The sealed forward test — the one backtest that cannot be overfit.

Every in-sample guardrail still runs on data the modeler has seen. The only fully honest
test is time itself: register a strategy (its config, hashed, with a timestamp), then score
it *only* on observations that arrive strictly after registration. `Registration` is the
receipt; `score_forward` refuses to score on anything dated at or before it. Wire the
post-registration feed to the ConvexPi Arena for a live paper track."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Registration:
    """A tamper-evident receipt for a strategy, fixing *what* was committed and *when*."""
    name: str
    config_hash: str
    registered_at: str          # ISO timestamp or a comparable date key
    config: dict = field(default_factory=dict)

    @property
    def token(self) -> str:
        return f"{self.name}:{self.config_hash[:12]}@{self.registered_at}"


def register_strategy(name: str, config: dict, when: str) -> Registration:
    """Commit a strategy config at time `when` (caller supplies the timestamp, so it is
    auditable and deterministic). The config is hashed; changing it later changes the hash."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    h = hashlib.sha256(blob).hexdigest()
    return Registration(name=name, config_hash=h, registered_at=str(when), config=dict(config))


def score_forward(registration: Registration, dates, returns, metric):
    """Score `metric(returns_after)` using only observations dated strictly after the
    registration time. Raises if nothing post-dates registration (no honest data yet)."""
    d = np.asarray(dates)
    r = np.asarray(returns, dtype=float)
    after = d > _as_key(registration.registered_at, d.dtype)
    if not after.any():
        raise ValueError(
            f"no data after registration {registration.registered_at!r}; "
            "the forward test has not accrued any genuinely out-of-sample observations yet"
        )
    return {"token": registration.token, "n_forward": int(after.sum()),
            "score": float(metric(r[after]))}


def _as_key(when, dtype):
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64(when)
    if np.issubdtype(dtype, np.number):
        return type(np.zeros(1, dtype)[0])(when)
    return when

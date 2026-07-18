"""A test set you can look at exactly once.

The single hardest habit to keep by willpower is *touch the test set once* (Rule 1's
operational form): every extra peek quietly turns the holdout into a validation set and
inflates the final number. `SealedHoldout` makes the rule mechanical — the second access
raises."""
from __future__ import annotations

from typing import Any


class HoldoutBreach(RuntimeError):
    """Raised when a sealed holdout is accessed more than the allowed number of times."""


class SealedHoldout:
    """Wrap held-out test data so it can be revealed only a fixed number of times.

    >>> h = SealedHoldout(X_test, y_test)
    >>> X, y = h.reveal("final evaluation of the LassoCV pipeline")   # ok, logged
    >>> h.reveal("let me just check one more thing")                  # raises HoldoutBreach

    Keep `max_reveals=1`. If you genuinely need a second look, that is a *design* decision
    you should make loudly (raise the limit in code), not a reflex.
    """

    def __init__(self, *payload: Any, max_reveals: int = 1):
        self._payload = payload
        self._max = int(max_reveals)
        self._log: list[str] = []

    @property
    def reveals(self) -> int:
        return len(self._log)

    @property
    def log(self) -> list[str]:
        return list(self._log)

    def reveal(self, reason: str):
        """Return the held-out data, recording why. Raises after `max_reveals`."""
        if not reason or not str(reason).strip():
            raise ValueError("state a reason for touching the test set")
        if self.reveals >= self._max:
            raise HoldoutBreach(
                f"test set already revealed {self.reveals} time(s) "
                f"(max {self._max}); prior reasons: {self._log}. "
                "Every extra peek makes this a validation set, not a test set."
            )
        self._log.append(str(reason))
        return self._payload if len(self._payload) != 1 else self._payload[0]

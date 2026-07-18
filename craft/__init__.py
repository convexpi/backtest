"""craft — the Rules of the Craft, made executable.

A thin discipline layer for backtests: it does not replace numpy/pandas/scikit-learn or
hide the methods. It wraps them with guardrails that make the honest-evaluation habits
mechanical instead of a matter of willpower —

  * a test set you can look at once (``SealedHoldout``),
  * time-aware splits by default (``walk_forward``, ``purged_walk_forward``),
  * a trial registry that feeds the multiple-testing correction (``TrialRegistry`` +
    ``deflated_sharpe``),
  * a reproducibility manifest (seed + data hash + versions + git sha),
  * and a six-question report card that stamps each answer to a numbered Rule.

Teach the habits by hand first; reach for ``craft`` once they are understood. Built to be
read and modified, in the spirit of ``finmlsim``.
"""
from .holdout import SealedHoldout, HoldoutBreach
from .splits import walk_forward, purged_walk_forward
from .registry import TrialRegistry, Trial
from .manifest import manifest, data_hash
from .report import card, ReportCard
from .rules import RULES, statement
from .metrics import (
    sharpe, max_drawdown, rank_ic, turnover, net_returns,
    probabilistic_sharpe, deflated_sharpe, expected_max_sharpe,
)

__version__ = "0.1.0"

__all__ = [
    "SealedHoldout", "HoldoutBreach",
    "walk_forward", "purged_walk_forward",
    "TrialRegistry", "Trial",
    "manifest", "data_hash",
    "card", "ReportCard",
    "RULES", "statement",
    "sharpe", "max_drawdown", "rank_ic", "turnover", "net_returns",
    "probabilistic_sharpe", "deflated_sharpe", "expected_max_sharpe",
]

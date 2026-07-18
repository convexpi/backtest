"""convexpi.backtest — a discipline-first backtesting framework.

Built around the book's *Rules of the Craft*: the honest-evaluation habits, made
mechanical, plus the statistics and machinery a serious backtest needs. It wraps the
standard stack (numpy/pandas/scikit-learn) rather than hiding it, and every guardrail
points back to a numbered Rule.

Core discipline
    SealedHoldout ................ a test set you can look at once
    walk_forward / purged_* ...... time-aware splits; no random k-fold
    TrialRegistry + deflated_sharpe  count the search; discount the winner
    manifest ..................... seed + data hash + versions + git sha
    card ......................... the six-question report card, per Rule

Validation & overfitting
    cv: purged_kfold, combinatorial_purged_split, cpcv_paths (a distribution of OOS paths)
    overfitting: pbo, haircut_sharpe, min_backtest_length, reality_check

Costs, risk, inference
    costs: CostModel, capacity                     bootstrap: bootstrap_ci, newey_west_tstat
    portfolio: vol_target, risk_parity, kelly_fraction, value_at_risk, expected_shortfall

Data, leakage, forward, governance
    data: asof_join, survivorship_free_universe
    leakage: shuffle_target_test, time_shuffle_test, target_leakage_scan, training_window_gradient
    forward: register_strategy, score_forward       (the sealed forward test)
    governance: backtest_card, AuditLedger

Teach the habits by hand first; reach for this once they are understood. Built to be read
and modified, in the spirit of ``finmlsim``.
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
from .cv import (
    purged_kfold, combinatorial_purged_split, cpcv_paths, n_backtest_paths,
)
from .overfitting import pbo, haircut_sharpe, min_backtest_length, reality_check
from .bootstrap import stationary_bootstrap_indices, bootstrap_ci, newey_west_tstat
from .costs import CostModel, capacity
from .portfolio import (
    vol_target, risk_parity, kelly_fraction, apply_constraints,
    value_at_risk, expected_shortfall,
)
from .leakage import (
    shuffle_target_test, time_shuffle_test, target_leakage_scan, training_window_gradient,
)
from .data import asof_join, survivorship_free_universe
from .forward import Registration, register_strategy, score_forward
from .governance import backtest_card, AuditLedger

__version__ = "0.2.0"

__all__ = [
    # core discipline
    "SealedHoldout", "HoldoutBreach", "walk_forward", "purged_walk_forward",
    "TrialRegistry", "Trial", "manifest", "data_hash", "card", "ReportCard",
    "RULES", "statement",
    "sharpe", "max_drawdown", "rank_ic", "turnover", "net_returns",
    "probabilistic_sharpe", "deflated_sharpe", "expected_max_sharpe",
    # validation & overfitting
    "purged_kfold", "combinatorial_purged_split", "cpcv_paths", "n_backtest_paths",
    "pbo", "haircut_sharpe", "min_backtest_length", "reality_check",
    # costs, risk, inference
    "CostModel", "capacity",
    "stationary_bootstrap_indices", "bootstrap_ci", "newey_west_tstat",
    "vol_target", "risk_parity", "kelly_fraction", "apply_constraints",
    "value_at_risk", "expected_shortfall",
    # data, leakage, forward, governance
    "asof_join", "survivorship_free_universe",
    "shuffle_target_test", "time_shuffle_test", "target_leakage_scan",
    "training_window_gradient",
    "Registration", "register_strategy", "score_forward",
    "backtest_card", "AuditLedger",
]

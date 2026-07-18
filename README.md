# convexpi.backtest — a discipline-first backtesting framework

A backtesting framework organized around the one thing that separates a real edge from a
lucky one: **honest evaluation**. It wraps the standard stack (numpy / pandas /
scikit-learn) rather than hiding it, and makes the honest habits mechanical.

Most backtesters make it *easy* to fool yourself — one train/test split, a gross Sharpe,
no record of how many things you tried. This one makes the honest habits mechanical and
the dishonest ones loud.

> **Teach the habits by hand first.** Reach for this once students understand *why* each
> guardrail exists — it automates the discipline, it does not substitute for learning it.
> Built to be read and modified, in the spirit of `finmlsim`.

## Install

```bash
pip install convexpi-backtest
```

```python
from convexpi import backtest as bt
```

## What's in it

**Core discipline** — the honest-evaluation habits, as objects:

| Piece | Enforces |
|---|---|
| `SealedHoldout` | Touch the test set once (raises on the second peek) |
| `walk_forward`, `purged_walk_forward` | Time-aware splits; no random k-fold |
| `TrialRegistry` + `deflated_sharpe` | Count every trial; discount the winner |
| `manifest` | Seed + data hash + versions + git sha on every result |
| `card(...)` | Answer the six diagnostic questions, ✓/✗/? |

**Validation & overfitting**
- `cv`: `purged_kfold`, `combinatorial_purged_split`, **`cpcv_paths`** — a *distribution* of
  out-of-sample paths, not one.
- `overfitting`: **`pbo`** (probability of backtest overfitting), `haircut_sharpe`
  (Harvey–Liu), `min_backtest_length`, `reality_check` (White).

**Costs, risk, inference**
- `costs`: `CostModel` (half-spread + commission + square-root impact), `capacity`.
- `portfolio`: `vol_target`, `risk_parity`, `kelly_fraction`, `apply_constraints`,
  `value_at_risk`, `expected_shortfall`.
- `bootstrap`: `bootstrap_ci` (stationary block bootstrap), `newey_west_tstat`.

**Data, leakage, forward, governance**
- `data`: `asof_join` (point-in-time), `survivorship_free_universe`.
- `leakage`: `shuffle_target_test`, `time_shuffle_test`, `target_leakage_scan`,
  **`training_window_gradient`** (learning vs recency).
- `forward`: **`register_strategy` + `score_forward`** — the *sealed forward test*: score a
  strategy only on data that arrives *after* it was registered. The one backtest that
  cannot be overfit; wire the post-registration feed to the ConvexPi Arena for a live
  paper track.
- `governance`: `backtest_card` (SR 11-7 flavored), `AuditLedger` (append-only trail).

## Quickstart

```python
import numpy as np
from convexpi import backtest as bt

# 1) A sealed test set — you may look once.
holdout = bt.SealedHoldout(X_test, y_test)

# 2) Search with time-aware splits and log every trial.
reg = bt.TrialRegistry(periods=252)
for cfg in grid:
    for train, val in bt.walk_forward(len(X_dev), n_splits=5):
        ...  # fit on train, score on val
    reg.log(cfg.name, returns=oos_returns(cfg), reason="lookback grid")

# 3) Reveal the test set once, cost the winner, and grade it.
Xte, yte = holdout.reveal("final evaluation of the selected model")
net = bt.CostModel(spread_bps=5, impact_coef=10).net(gross, weights, adv_fraction=0.05)
print(bt.card(net_returns=net, gross_returns=gross, baseline_returns=baseline,
              registry=reg, holdout=holdout, breadth=40,
              manifest=bt.manifest(seed=7, data=prices)))

# 4) Stress the selection itself.
print("PBO:", bt.pbo(candidate_returns_matrix)["pbo"])           # ~0.5+ => overfit
paths = bt.cpcv_paths(len(y), predict, n_groups=6, n_test_groups=2)
print("OOS Sharpe dispersion:", np.nanstd([bt.sharpe(p) for p in paths]))
```

## License

MIT © Shane Conway. A companion to the `finmlsim` (simulators) and `convexpi.arena`
(live exchange) packages.

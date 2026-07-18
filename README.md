# craft — the Rules of the Craft, made executable

A thin **discipline layer** for financial backtests. It does **not** replace
numpy / pandas / scikit-learn, and it does not hide the methods. It wraps them with
guardrails that make the honest-evaluation habits *mechanical* instead of a matter of
willpower — the habits from Chapter 1 of *Financial Machine Learning and AI*:

> set seeds · version your data · **touch the test set once** · **log every search** ·
> **walk-forward by default** · write the reason down

`craft` is those habits turned into objects, each pointing back to a numbered **Rule of
the Craft**. It is a teaching aid in the spirit of `finmlsim`: small, dependency-light
(numpy only), and *built to be read and modified*.

> **Teach the habits by hand first.** Reach for `craft` once students understand *why*
> each guardrail exists — it automates the discipline, it does not substitute for learning it.

## Install

```bash
pip install convexpi-craft          # numpy only
```

## What it gives you

| Guardrail | Habit it enforces | Rule |
|---|---|---|
| `SealedHoldout` | Touch the test set once (raises on the second peek) | 1 |
| `walk_forward`, `purged_walk_forward` | Time-aware splits by default; no random k-fold | 3, 5 |
| `TrialRegistry` + `deflated_sharpe` | Count every trial; discount the winner for the search | 14 |
| `net_returns`, report card | Report net of costs, against a baseline | 6, 14 |
| `manifest` | Seed + data hash + versions + git sha on every result | 0, 2 |
| `card(...)` | Answer Chapter 1's six questions, ✓/✗/? per Rule | — |

## Quickstart

```python
import numpy as np, craft

rng = np.random.default_rng(7)

# 1) A sealed test set — you may look once.
X_test, y_test = ...                     # your held-out block
holdout = craft.SealedHoldout(X_test, y_test)

# 2) Time-aware splits for the search (no shuffling).
for train_idx, val_idx in craft.walk_forward(len(X), n_splits=5):
    ...

# 3) Log every configuration you try — the count feeds the deflated Sharpe.
reg = craft.TrialRegistry(periods=252)
for cfg in grid:
    reg.log(cfg.name, returns=backtest(cfg), reason="grid over lookback windows")

# 4) Reveal the test set once, evaluate the winner, and grade the whole thing.
Xte, yte = holdout.reveal("final evaluation of the selected model")
net = craft.net_returns(gross_returns, weights, cost_bps=10)
report = craft.card(
    net_returns=net, gross_returns=gross_returns, baseline_returns=baseline,
    registry=reg, holdout=holdout, breadth=40,
    manifest=craft.manifest(seed=7, data=prices),
)
print(report)      # ✓/✗/? for each of the six questions, stamped to a Rule
```

The winner of a 300-config grid on noise will fail the *"paid for the search"* check —
which is the point.

## Roadmap (toward a proper framework)

The first cut is the six-habit core. A fuller build would add, roughly outward:

- **Multiple-testing statistics:** PBO via combinatorial-purged CV, Harvey–Liu haircut,
  White's Reality Check / Hansen's SPA, minimum backtest length.
- **Validation:** combinatorial purged CV (a *distribution* of OOS paths), regime/stress
  windows.
- **Point-in-time data:** bitemporal as-of joins, survivorship-free universes,
  corporate-action provenance.
- **Costs & capacity:** square-root impact, borrow/financing, turnover → capacity.
- **Inference on dependent data:** stationary block-bootstrap CIs (from `finmlsim`),
  Newey–West IC t-stats.
- **Leakage detectors:** shuffle-test, target-in-features scan, the training-window
  gradient (learning vs recency, Ch 18).
- **Governance & the honest endgame:** a "backtest card" (SR 11-7 flavored), and a
  sealed *forward* test — register a strategy, then score it only on data that arrives
  *after* registration, wired to the ConvexPi Arena.

## License

MIT © Shane Conway. Companion to *Financial Machine Learning and AI* and to `finmlsim`.

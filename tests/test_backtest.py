"""Tests for convexpi.backtest."""
import numpy as np
import pytest

import convexpi.backtest as bt


# ---- sealed holdout -------------------------------------------------------
def test_holdout_reveals_once_then_raises():
    h = bt.SealedHoldout(np.arange(10), max_reveals=1)
    out = h.reveal("final evaluation")
    assert np.array_equal(out, np.arange(10))
    assert h.reveals == 1
    with pytest.raises(bt.HoldoutBreach):
        h.reveal("just one more peek")


def test_holdout_requires_reason():
    h = bt.SealedHoldout(1, 2)
    with pytest.raises(ValueError):
        h.reveal("")


def test_holdout_returns_tuple_for_multiple_payload():
    h = bt.SealedHoldout(np.arange(3), np.arange(3))
    x, y = h.reveal("eval")
    assert len(x) == 3 and len(y) == 3


# ---- splits ---------------------------------------------------------------
def test_walk_forward_is_causal():
    for tr, te in bt.walk_forward(100, n_splits=4):
        assert tr.max() < te.min(), "training must precede test (no look-ahead)"


def test_purged_walk_forward_purges_and_embargoes():
    for tr, te in bt.purged_walk_forward(120, n_splits=4, embargo=3, label_horizon=2):
        assert tr.max() < te.min()
        # no training row within the embargo window after the test block
        assert not ((tr > te[-1]) & (tr <= te[-1] + 3)).any()


# ---- metrics --------------------------------------------------------------
def test_sharpe_and_drawdown():
    rng = np.random.default_rng(0)
    r = 0.0004 + 0.01 * rng.standard_normal(2520)
    assert np.isfinite(bt.sharpe(r))
    assert bt.max_drawdown(r) <= 0.0


def test_rank_ic_perfect_monotone():
    x = np.arange(50, dtype=float)
    assert bt.rank_ic(x, 2 * x + 1) == pytest.approx(1.0)
    assert bt.rank_ic(x, -x) == pytest.approx(-1.0)


def test_deflated_sharpe_penalizes_search():
    # same observed Sharpe is less credible after more trials
    d_few = bt.deflated_sharpe(0.10, 0.05, n_trials=5, n_obs=1000)
    d_many = bt.deflated_sharpe(0.10, 0.05, n_trials=500, n_obs=1000)
    assert 0.0 <= d_many <= d_few <= 1.0


# ---- registry -------------------------------------------------------------
def test_registry_counts_and_deflates():
    rng = np.random.default_rng(1)
    reg = bt.TrialRegistry(periods=252)
    for i in range(200):  # a big null search
        reg.log(f"cfg{i}", returns=0.01 * rng.standard_normal(1000),
                reason="grid over lookback")
    assert reg.n_trials == 200
    # best-of-200 on pure noise should not be credible
    assert reg.deflated_best(n_obs=1000) < 0.95


def test_registry_requires_reason():
    reg = bt.TrialRegistry()
    with pytest.raises(ValueError):
        reg.log("x", sharpe=1.0)


# ---- manifest -------------------------------------------------------------
def test_manifest_and_data_hash_stable():
    x = np.arange(100)
    assert bt.data_hash(x) == bt.data_hash(np.arange(100))
    assert bt.data_hash(x) != bt.data_hash(np.arange(101))
    m = bt.manifest(seed=42, data=x, when="2026-07-18T00:00:00+00:00")
    assert m["seed"] == 42 and m["data_hash"] and m["timestamp"].startswith("2026")


# ---- report card ----------------------------------------------------------
def test_card_flags_unpaid_search_and_unknowns():
    rng = np.random.default_rng(2)
    gross = 0.0003 + 0.01 * rng.standard_normal(1000)
    net = bt.net_returns(gross, np.zeros((1000, 1)), cost_bps=10)
    base = 0.01 * rng.standard_normal(1000)
    reg = bt.TrialRegistry()
    for i in range(300):
        reg.log(f"c{i}", returns=0.01 * rng.standard_normal(1000), reason="search")
    h = bt.SealedHoldout(net)
    m = bt.manifest(seed=0, data=gross, when="2026-07-18T00:00:00+00:00")
    rc = bt.card(net_returns=net, gross_returns=gross, baseline_returns=base,
                    registry=reg, holdout=h, manifest=m, breadth=50)
    assert isinstance(str(rc), str)
    # the big null search should fail the "paid for the search" check
    marks = {q: mk for q, mk, _, _ in rc.checks}
    assert marks["Paid for the search (deflated Sharpe)?"] == "✗"

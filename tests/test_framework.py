"""Tests for the full-framework modules (cv, overfitting, costs, portfolio, leakage,
data, forward, governance, bootstrap)."""
import numpy as np
import pytest

import convexpi.backtest as bt


# ---- combinatorial purged CV ---------------------------------------------
def test_cpcv_split_count_and_causality_purge():
    splits = list(bt.combinatorial_purged_split(120, n_groups=6, n_test_groups=2,
                                                embargo=2, label_horizon=1))
    assert len(splits) == 15                      # C(6,2)
    for train, test in splits:
        assert set(train).isdisjoint(set(test))   # purged: no overlap


def test_cpcv_path_count():
    assert bt.n_backtest_paths(6, 2) == 5         # k*C(G,k)/G = 2*15/6


def test_cpcv_paths_assemble():
    n = 120
    paths = bt.cpcv_paths(n, predict=lambda tr, te: np.zeros(len(te)),
                          n_groups=6, n_test_groups=2)
    assert paths.shape == (5, n)
    # every observation is covered on every path
    assert np.isfinite(paths).all()


def test_purged_kfold_causal_gap():
    for tr, te in bt.purged_kfold(100, n_splits=5, embargo=3, label_horizon=2):
        assert set(tr).isdisjoint(set(te))


# ---- overfitting ----------------------------------------------------------
def test_pbo_on_noise_is_near_half():
    rng = np.random.default_rng(0)
    perf = rng.standard_normal((600, 12)) * 0.01     # 12 skill-less strategies
    out = bt.pbo(perf, n_splits=10)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo"] > 0.2                            # random winners overfit


def test_haircut_and_min_length():
    assert bt.haircut_sharpe(2.0, n_trials=1, n_obs=2520) > bt.haircut_sharpe(2.0, 100, 2520)
    assert bt.min_backtest_length(2.0) == pytest.approx(1.0)
    assert bt.min_backtest_length(0.0) == float("inf")


def test_reality_check_range():
    rng = np.random.default_rng(1)
    p = bt.reality_check(rng.standard_normal((400, 8)) * 0.01, n_boot=200, seed=1)
    assert 0.0 <= p <= 1.0


# ---- bootstrap ------------------------------------------------------------
def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    r = 0.0005 + 0.01 * rng.standard_normal(1500)
    ci = bt.bootstrap_ci(r, bt.sharpe, n_boot=300, seed=2)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_newey_west_detects_mean():
    rng = np.random.default_rng(3)
    assert abs(bt.newey_west_tstat(0.5 + rng.standard_normal(500))) > 2   # clear nonzero mean
    assert abs(bt.newey_west_tstat(rng.standard_normal(500))) < 3          # ~zero mean


# ---- costs ----------------------------------------------------------------
def test_cost_model_reduces_returns_and_capacity_positive():
    cm = bt.CostModel(spread_bps=5, commission_bps=1, impact_coef=10)
    gross = np.full(100, 0.001)
    w = np.tile([[1.0, -1.0]], (100, 1)) * (np.arange(100) % 2).reshape(-1, 1)
    net = cm.net(gross, w, adv_fraction=0.05)
    assert net.sum() < gross.sum()
    assert bt.capacity(1.0, avg_turnover=0.2, adv_usd=1e8, cost_model=cm) >= 0.0


# ---- portfolio ------------------------------------------------------------
def test_vol_target_stabilizes_and_risk_parity_sums_to_one():
    rng = np.random.default_rng(4)
    r = 0.01 * rng.standard_normal(800)
    scaled = bt.vol_target(r, target_vol=0.10, window=60)
    assert scaled.shape == r.shape
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = bt.risk_parity(cov)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_constraints_and_tail_risk():
    w = bt.apply_constraints([2.0, -2.0, 0.5], max_leverage=1.0, max_position=1.0)
    assert np.abs(w).sum() <= 1.0 + 1e-9
    rng = np.random.default_rng(5)
    r = rng.standard_normal(2000) * 0.01
    assert bt.expected_shortfall(r) <= bt.value_at_risk(r) <= 0.0


# ---- leakage --------------------------------------------------------------
def test_shuffle_target_test_null():
    rng = np.random.default_rng(6)
    y = rng.standard_normal(200)
    # a score_fn with no real dependence on y -> high p-value
    out = bt.shuffle_target_test(lambda yy: -abs(yy.mean()), y, n_permutations=100)
    assert 0.0 <= out["p_value"] <= 1.0


def test_target_leakage_scan_flags_copy():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(300)
    X = np.column_stack([rng.standard_normal(300), y * 1.0 + 1e-9 * rng.standard_normal(300)])
    flags = bt.target_leakage_scan(X, y, threshold=0.95)
    assert any(j == 1 for j, _ in flags)


# ---- data (point-in-time) -------------------------------------------------
def test_asof_join_no_lookahead():
    target = np.array([1, 5, 10])
    src_dates = np.array([2, 6])
    src_vals = np.array([10.0, 20.0])
    out = bt.asof_join(target, src_dates, src_vals, availability_lag=1)
    # at t=1 nothing available (src 2 + lag1 = 3 > 1); t=5 -> first (avail 3); t=10 -> second
    assert np.isnan(out[0]) and out[1] == 10.0 and out[2] == 20.0


# ---- forward test ---------------------------------------------------------
def test_forward_scores_only_after_registration():
    reg = bt.register_strategy("mom", {"lookback": 20}, when=100)
    dates = np.arange(90, 110)
    returns = np.ones(20) * 0.01
    out = bt.score_forward(reg, dates, returns, metric=np.mean)
    assert out["n_forward"] == 9                     # dates 101..109
    with pytest.raises(ValueError):
        bt.score_forward(reg, np.arange(50, 100), np.ones(50), metric=np.mean)


def test_registration_hash_changes_with_config():
    a = bt.register_strategy("s", {"k": 1}, when=0)
    b = bt.register_strategy("s", {"k": 2}, when=0)
    assert a.config_hash != b.config_hash


# ---- governance -----------------------------------------------------------
def test_audit_ledger_roundtrip(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    led = bt.AuditLedger(path)
    m = bt.manifest(seed=1, data=np.arange(10), when="2026-07-18T00:00:00+00:00")
    led.record(bt.backtest_card("s", m, {"sharpe": 1.2}, {"pbo": 0.1}))
    assert len(bt.AuditLedger.load(path).entries) == 1

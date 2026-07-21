"""Regression tests for causal execution and scientifically valid splits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from agent.reward import RiskAwareReward
from alpha.feature_engineering import AlphaFeatureEngine
from alpha.temporal_features import TemporalFeatureBuilder
from env.trading_env import NSETradingEnv


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["market"]["tickers"] = ["TEST.NS"]
    cfg["llm"]["sentiment_dim"] = 0
    cfg["alpha"]["obs_dim"] = 0
    cfg["meta_model"]["enabled"] = False
    cfg["patchtst"]["enabled"] = False
    cfg["gnn"]["enabled"] = False
    cfg["vae_regime"]["enabled"] = False
    cfg["llm"]["debate"]["enabled"] = False
    cfg["risk_profiles"]["balanced"]["stop_loss_pct"] = 0.99
    return cfg


def _make_env(config, closes=None, terminate_on_stop=True):
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = np.asarray(closes or [100, 101, 103, 102, 105, 106], dtype=float)
    frame = pd.DataFrame(
        {
            "Close": close,
            "feature": np.arange(10, 16, dtype=float),
        },
        index=dates,
    )
    return NSETradingEnv(
        data={"TEST.NS": frame},
        feature_names=["feature"],
        config=config,
        risk_profile="balanced",
        term="medium",
        terminate_on_stop=terminate_on_stop,
    )


def test_observation_is_lagged_one_bar_from_execution(config):
    env = _make_env(config)
    obs, info = env.reset()

    # Reset is positioned at bar 1, but the market observation is from bar 0.
    assert env._current_step == 1
    assert obs[0] == pytest.approx(10.0)

    _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert info["signal_date"][:10] == "2024-01-01"
    assert info["execution_date"][:10] == "2024-01-02"
    assert info["date"][:10] == "2024-01-03"


def test_weights_are_marked_to_market_after_price_move(config):
    env = _make_env(config)
    env.reset()
    env.step(np.ones(1, dtype=np.float32))

    marked_weight = (
        env.positions[0] * env._get_prices(env._current_step)[0]
        / env.portfolio_value
    )
    assert env.weights[0] == pytest.approx(marked_weight)
    assert env.weights[0] != pytest.approx(env.max_delta)


def test_cash_equity_mode_rejects_short_positions(config):
    config["env"]["allow_short"] = False
    env = _make_env(config)
    env.reset()
    env.step(-np.ones(1, dtype=np.float32))
    assert env.weights[0] == pytest.approx(0.0)
    assert env.positions[0] == pytest.approx(0.0)


def test_eval_stop_loss_liquidates_with_cost_and_stays_in_cash(config):
    config["risk_profiles"]["balanced"]["stop_loss_pct"] = 0.001
    env = _make_env(
        config,
        closes=[100, 100, 80, 79, 78, 77],
        terminate_on_stop=False,
    )
    env.reset()

    _, _, _, truncated, info = env.step(np.ones(1, dtype=np.float32))
    assert info["stop_triggered"] is True
    assert truncated is False
    assert env._stopped_out is True
    assert np.allclose(env.positions, 0.0)
    assert np.allclose(env.weights, 0.0)
    assert info["transaction_cost"] > 0

    cash = env.cash
    env.step(np.ones(1, dtype=np.float32))
    assert env.cash == pytest.approx(cash)
    assert env.portfolio_value == pytest.approx(cash)


def test_alpha_alignment_bonus_uses_pre_action_weights(config):
    config["env"]["alpha_reward_bonus"] = 0.05
    config["env"]["alpha_conviction_threshold"] = 0.5
    aligned = SimpleNamespace(
        direction=np.ones(1, dtype=np.float32),
        conviction=np.ones(1, dtype=np.float32),
    )

    with_alpha = RiskAwareReward(config, "balanced")
    without_alpha = RiskAwareReward(config, "balanced")
    for reward in (with_alpha, without_alpha):
        reward.compute(
            step_return=0.0,
            portfolio_values=[1_000_000],
            current_weights=np.zeros(1),
        )

    kwargs = {
        "step_return": 0.0,
        "portfolio_values": [1_000_000, 1_000_000],
        "current_weights": np.array([0.05]),
    }
    rewarded = with_alpha.compute(**kwargs, alpha_signal=aligned)
    baseline = without_alpha.compute(**kwargs, alpha_signal=None)
    assert rewarded > baseline


def test_cross_sectional_ranks_do_not_change_with_future_rows(config):
    dates = pd.bdate_range("2024-01-01", periods=4)
    data = {}
    for ticker, offset in (("A", 0.0), ("B", 1.0), ("C", 2.0)):
        data[ticker] = pd.DataFrame(
            {
                "ret_1d": np.array([1, 2, 3, 4], dtype=float) + offset,
                "ret_20d": np.array([4, 3, 2, 1], dtype=float) - offset,
                "vol_20d": np.array([1, 2, 1, 2], dtype=float) + offset,
            },
            index=dates,
        )

    engine = AlphaFeatureEngine(config)
    full = engine._compute_cross_stock_features(
        {ticker: frame.copy() for ticker, frame in data.items()}
    )
    truncated = engine._compute_cross_stock_features(
        {ticker: frame.iloc[:2].copy() for ticker, frame in data.items()}
    )

    for ticker in data:
        for feature in ("ret_rank", "vol_rank", "mom_rank"):
            assert full[ticker].loc[dates[0], feature] == pytest.approx(
                truncated[ticker].loc[dates[0], feature]
            )


def test_temporal_validation_split_is_chronological_across_tickers(config):
    config["patchtst"]["forward_horizon"] = 2
    builder = TemporalFeatureBuilder(config, ["feature"])
    dates = pd.bdate_range("2024-01-01", periods=10)
    meta = [
        (ticker, date)
        for ticker in ("A", "B")
        for date in dates
    ]
    sequences = np.arange(len(meta), dtype=np.float32).reshape(-1, 1, 1)
    labels = np.arange(len(meta), dtype=np.float32)

    x_train, _, x_val, _ = builder.build_validation_split(
        sequences, labels, meta, val_fraction=0.2
    )
    train_ids = x_train[:, 0, 0].astype(int)
    val_ids = x_val[:, 0, 0].astype(int)
    train_meta = [meta[i] for i in train_ids]
    val_meta = [meta[i] for i in val_ids]

    assert {ticker for ticker, _ in train_meta} == {"A", "B"}
    assert {ticker for ticker, _ in val_meta} == {"A", "B"}
    assert max(date for _, date in train_meta) < min(
        date for _, date in val_meta
    )


def test_fold_static_channel_signals_are_cached_across_episodes(config):
    config["alpha"]["obs_dim"] = 2
    calls = {"count": 0}

    class Signal:
        def to_obs_vector(self):
            return np.array([0.5, 0.8], dtype=np.float32)

    def alpha_fn(date):
        calls["count"] += 1
        return Signal()

    env = _make_env(config)
    env.alpha_fn = alpha_fn
    env._channel_enabled["alpha"] = True

    for _ in range(2):
        env.reset()
        done = truncated = False
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(
                np.zeros(1, dtype=np.float32)
            )

    # Five distinct lagged signal dates are traversed; the second episode
    # should be served entirely from cache.
    assert calls["count"] == 5

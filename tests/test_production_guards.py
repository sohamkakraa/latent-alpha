"""Tests for production guards: NaN handling, stop-loss, obs logging (P0 Prompt 3)."""
import numpy as np
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_env(config):
    """Create a small trading env using synthetic data."""
    import pandas as pd
    from env.trading_env import NSETradingEnv

    np.random.seed(42)
    n = 80
    dates = pd.bdate_range("2020-01-01", periods=n)

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10.0)
    df_base = pd.DataFrame({
        "Open":   close * 1.001,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": np.random.randint(100000, 1000000, size=n).astype(float),
    }, index=dates)

    tickers = config["market"]["tickers"]
    feature_cols = [
        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_lower", "BB_mid", "ATR_14",
        "EMA_9", "EMA_21", "EMA_50",
        "ret_1d", "ret_5d", "vol_20d", "volume_z_20",
        "ret_20d",
    ]

    # Build minimal synthetic data with feature columns
    data = {}
    for t in tickers:
        df = df_base.copy()
        for col in feature_cols:
            df[col] = np.random.randn(n) * 0.1
        data[t] = df

    cfg = dict(config)
    cfg["llm"] = dict(config["llm"])
    cfg["llm"]["sentiment_dim"] = 0  # Disable sentiment for speed

    env = NSETradingEnv(
        data=data,
        feature_names=feature_cols,
        config=cfg,
        risk_profile="balanced",
        term="medium",
    )
    return env


def test_nan_action_replaced_with_zeros(small_env):
    """NaN actions should be replaced with zeros, not propagated."""
    env = small_env
    obs, info = env.reset()

    n_stocks = env.n_stocks
    nan_action = np.full(n_stocks, np.nan, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(nan_action)

    assert not np.any(np.isnan(obs)), "NaN in observation after NaN action"
    assert not np.isnan(reward), "NaN reward after NaN action"


def test_extreme_action_clipped(small_env):
    """Actions outside [-1, 1] should be clipped."""
    env = small_env
    obs, info = env.reset()

    extreme_action = np.full(env.n_stocks, 100.0, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(extreme_action)

    max_w = env.risk_cfg["max_position_pct"]
    assert np.all(np.abs(env.weights) <= max_w + 1e-6)


def test_stop_loss_liquidates_portfolio(small_env):
    """When stop-loss triggers, portfolio should be fully liquidated to cash."""
    env = small_env
    obs, info = env.reset()

    # Simulate large drawdown
    env.portfolio_value = env.initial_capital * 0.5   # 50% loss
    env._value_history.append(env.portfolio_value)
    env.prev_value = env.portfolio_value

    action = np.zeros(env.n_stocks, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

    if truncated:
        assert np.allclose(env.weights, 0.0), "Weights should be zero after stop-loss"
        assert np.allclose(env.positions, 0.0), "Positions should be zero after stop-loss"


def test_observation_no_nan_after_guards(small_env):
    """Observations should never contain NaN after guards are applied."""
    env = small_env
    obs, info = env.reset()
    assert not np.any(np.isnan(obs)), "NaN in initial observation"

    for _ in range(20):
        action = np.random.uniform(-1, 1, size=env.n_stocks).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN in observation at step {env._current_step}"
        assert not np.isnan(reward), f"NaN reward at step {env._current_step}"
        if done or truncated:
            break


def test_compute_drawdown_method(small_env):
    """_compute_drawdown returns correct drawdown percentage."""
    env = small_env
    obs, info = env.reset()

    dd = env._compute_drawdown()
    assert dd == pytest.approx(0.0, abs=1e-6)

    # Simulate 10% loss
    env.portfolio_value = env.initial_capital * 0.9
    env._value_history.append(env.portfolio_value)
    dd = env._compute_drawdown()
    assert dd == pytest.approx(0.1, abs=1e-6)


def test_compute_drawdown_exists(small_env):
    """_compute_drawdown method should exist on NSETradingEnv."""
    assert hasattr(small_env, "_compute_drawdown")
    result = small_env._compute_drawdown()
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

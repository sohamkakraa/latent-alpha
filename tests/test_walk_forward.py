"""Tests for walk-forward cache safety and retrain mode contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from backtest.runner import WalkForwardBacktester
from env.data_loader import NSEDataLoader


class _OneStepBacktestEnv(Env):
    """Minimal env that completes in one step."""

    def __init__(self, n_stocks: int):
        super().__init__()
        self.n_stocks = n_stocks
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_stocks,), dtype=np.float32
        )
        self._done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._done = False
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._done = True
        info = {
            "portfolio_value": 1_000_000.0,
            "prev_value": 1_000_000.0,
            "weights": [0.0] * self.n_stocks,
            "date": "2021-01-01",
        }
        return np.zeros(3, dtype=np.float32), 0.0, True, False, info


class _FakeAgent:
    """Agent stub for lightweight _run_fold tests."""

    def __init__(self, n_stocks: int):
        self.n_stocks = n_stocks

    def train(self, train_env, eval_env=None, resume=False, total_timesteps=None):
        return None

    def load(self, raw_env=None):
        return None

    def predict(self, obs, deterministic=True):
        return np.zeros((1, self.n_stocks), dtype=np.float32), None

    def _checkpoint_exists(self):
        return True


class _TrackingDummyVecEnv(DummyVecEnv):
    """Records whether rollout used this wrapped/normalized environment."""

    def __init__(self, n_stocks: int):
        super().__init__([lambda: _OneStepBacktestEnv(n_stocks)])
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self):
        self.reset_calls += 1
        return super().reset()

    def step_wait(self):
        self.step_calls += 1
        return super().step_wait()


class _FakeModel:
    def __init__(self, env):
        self._env = env

    def get_env(self):
        return self._env


@pytest.fixture
def config(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    return cfg


def test_retrain_updates_cache_fold_id_each_fold(config):
    backtester = WalkForwardBacktester(config, risk_profile="balanced", term="medium")
    n_stocks = len(config["market"]["tickers"])

    backtester._make_agent = lambda *args: _FakeAgent(n_stocks)  # type: ignore[method-assign]

    observed_fold_ids: list[tuple[str, str]] = []

    def fake_build_env(
        *,
        risk_profile,
        term,
        train_start,
        train_end,
        sentiment_fn=None,
        sentiment_fn_eval=None,
    ):
        backtester.selector._trained_fns = {  # type: ignore[attr-defined]
            "alpha_fn": object(),
            "temporal_fn": object(),
            "gnn_fn": object(),
            "regime_fn": object(),
            "fold_id": (train_start, train_end),
        }
        dummy = DummyVecEnv([lambda: _OneStepBacktestEnv(n_stocks)])
        return dummy, dummy

    def fake_build_backtest_env(*, risk_profile, term, start, end, train_start, train_end, sentiment_fn=None, **kwargs):
        observed_fold_ids.append(backtester.selector._trained_fns["fold_id"])  # type: ignore[index]
        assert backtester.selector._trained_fns["fold_id"] == (train_start, train_end)  # type: ignore[index]
        return _OneStepBacktestEnv(n_stocks)

    backtester.selector.build_env = fake_build_env  # type: ignore[method-assign]
    backtester.selector.build_backtest_env = fake_build_backtest_env  # type: ignore[method-assign]

    backtester._run_fold(  # type: ignore[arg-type]
        fold_idx=0,
        train_start="2018-01-01",
        train_end="2021-01-01",
        test_start="2021-01-01",
        test_end="2021-07-01",
        retrain=True,
    )
    backtester._run_fold(  # type: ignore[arg-type]
        fold_idx=1,
        train_start="2018-07-01",
        train_end="2021-07-01",
        test_start="2021-07-01",
        test_end="2022-01-01",
        retrain=True,
    )

    assert len(observed_fold_ids) == 2
    assert observed_fold_ids[0] != observed_fold_ids[1]


def test_run_retrain_false_raises_clear_error(config):
    backtester = WalkForwardBacktester(config, risk_profile="balanced", term="medium")
    with pytest.raises(NotImplementedError, match="retrain_each_fold=False"):
        backtester.run(retrain_each_fold=False)


def test_fold_rollout_uses_agent_normalized_environment(config):
    backtester = WalkForwardBacktester(
        config, risk_profile="balanced", term="medium"
    )
    n_stocks = len(config["market"]["tickers"])
    wrapped_env = _TrackingDummyVecEnv(n_stocks)
    fake_agent = _FakeAgent(n_stocks)
    fake_agent.model = _FakeModel(wrapped_env)
    backtester._make_agent = lambda *args: fake_agent  # type: ignore[method-assign]

    dummy_train = DummyVecEnv([lambda: _OneStepBacktestEnv(n_stocks)])
    backtester.selector.build_env = lambda **kwargs: (  # type: ignore[method-assign]
        dummy_train,
        dummy_train,
    )
    backtester.selector.build_backtest_env = lambda **kwargs: (  # type: ignore[method-assign]
        _OneStepBacktestEnv(n_stocks)
    )

    backtester._run_fold(
        fold_idx=0,
        train_start="2018-01-01",
        train_end="2021-01-01",
        test_start="2021-01-01",
        test_end="2021-07-01",
        retrain=True,
    )

    assert wrapped_env.reset_calls == 1
    assert wrapped_env.step_calls == 1


def test_backtest_rebinds_trained_channels_to_oos_panel(config):
    from strategy.term_selector import TermSelector

    dates = pd.bdate_range("2020-12-01", "2021-07-01")
    feature_names = NSEDataLoader(config).get_feature_names()
    data = {}
    for ticker in config["market"]["tickers"]:
        frame = pd.DataFrame(
            {
                "Close": np.linspace(100.0, 120.0, len(dates)),
                **{
                    feature: np.linspace(-1.0, 1.0, len(dates))
                    for feature in feature_names
                },
            },
            index=dates,
        )
        data[ticker] = frame

    cache_key = "2018-01-01_2021-07-01_medium"
    selector = TermSelector(config, data_cache={cache_key: data})
    selector._trained_fns = {
        "fold_id": ("2018-01-01", "2021-01-01"),
    }
    observed = {}

    def bind(inference_data):
        observed["max_date"] = max(next(iter(inference_data.values())).index)
        return None, None, None, None

    selector._bind_trained_channel_fns = bind  # type: ignore[method-assign]
    selector.build_backtest_env(
        risk_profile="balanced",
        term="medium",
        start="2021-01-01",
        end="2021-07-01",
        train_start="2018-01-01",
        train_end="2021-01-01",
    )

    assert observed["max_date"] >= pd.Timestamp("2021-06-30")

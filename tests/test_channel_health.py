"""Tests for per-channel health tracking and dead-channel guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from env.trading_env import NSETradingEnv


class _AlwaysFailAlpha:
    """Stub alpha channel that always raises."""

    def __call__(self, date):
        raise RuntimeError("alpha unavailable")


@pytest.fixture
def base_config():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"] = dict(cfg["llm"])
    cfg["llm"]["sentiment_dim"] = 0
    cfg["meta_model"] = dict(cfg.get("meta_model", {}))
    cfg["meta_model"]["enabled"] = False
    cfg["alpha"] = dict(cfg.get("alpha", {}))
    cfg["alpha"]["enabled"] = True
    cfg["alpha"]["obs_dim"] = 2
    return cfg


def _make_env(cfg: dict, fail_threshold):
    dates = pd.bdate_range("2020-01-01", periods=260)
    close = np.linspace(100.0, 120.0, len(dates), dtype=np.float32)
    df = pd.DataFrame(
        {
            "Close": close,
            "f1": np.linspace(-1.0, 1.0, len(dates), dtype=np.float32),
        },
        index=dates,
    )
    cfg = dict(cfg)
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["fail_on_dead_channel"] = fail_threshold
    return NSETradingEnv(
        data={"TEST.NS": df},
        feature_names=["f1"],
        config=cfg,
        risk_profile="balanced",
        term="medium",
        alpha_fn=_AlwaysFailAlpha(),
    )


def test_dead_channel_guard_and_episode_summary_log(base_config, caplog):
    caplog.set_level("INFO")

    # Guarded run: always-failing alpha should trip the 200-step dead-channel guard.
    env_guarded = _make_env(base_config, fail_threshold=0.5)
    env_guarded.reset()
    with pytest.raises(RuntimeError, match="Dead channel guard triggered"):
        for _ in range(250):
            env_guarded.step(np.zeros(env_guarded.n_stocks, dtype=np.float32))

    # Non-guarded run: same failure should degrade gracefully and log summary at episode end.
    env_no_guard = _make_env(base_config, fail_threshold=None)
    env_no_guard.reset()
    done = truncated = False
    while not (done or truncated):
        _, _, done, truncated, _ = env_no_guard.step(
            np.zeros(env_no_guard.n_stocks, dtype=np.float32)
        )

    assert "Channel health: alpha=0%" in caplog.text

"""Unit tests for agent/reward.py — risk-aware reward components."""
import numpy as np
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def reward_fn(config):
    from agent.reward import RiskAwareReward
    return RiskAwareReward(config, risk_profile="balanced")


class TestRiskAwareReward:
    """Tests for RiskAwareReward.compute()."""

    def test_positive_return_positive_reward(self, reward_fn):
        """Positive step return should contribute positively to reward."""
        reward_fn.reset()
        r = reward_fn.compute(
            step_return=0.01,
            portfolio_values=[1_000_000, 1_010_000],
        )
        assert r > 0, f"Positive return gave negative reward: {r}"

    def test_negative_return_negative_reward(self, reward_fn):
        """Large negative return should give negative reward."""
        reward_fn.reset()
        r = reward_fn.compute(
            step_return=-0.05,
            portfolio_values=[1_000_000, 950_000],
        )
        assert r < 0, f"Large negative return gave positive reward: {r}"

    def test_drawdown_penalty_increases_with_drawdown(self, reward_fn):
        """Larger drawdowns should produce more negative reward."""
        reward_fn.reset()
        r_small = reward_fn.compute(
            step_return=-0.01,
            portfolio_values=[1_000_000, 990_000],
        )

        reward_fn.reset()
        r_large = reward_fn.compute(
            step_return=-0.05,
            portfolio_values=[1_000_000, 950_000],
        )

        assert r_large < r_small, "Larger drawdown should give worse reward"

    def test_turnover_penalty_quadratic(self, reward_fn):
        """Large weight changes should be penalised more than small ones."""
        w1 = np.array([0.1] * 10, dtype=np.float32)

        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        w2_small = w1 + 0.01
        r_small = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_small,
        )

        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        w2_large = w1 + 0.1
        r_large = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_large,
        )

        assert r_large < r_small, (
            f"Large rebalance ({r_large:.4f}) should be penalised more than "
            f"small rebalance ({r_small:.4f})"
        )

    def test_holding_bonus_for_stable_positions(self, reward_fn):
        """Keeping positions stable should give a bonus."""
        w1 = np.array([0.1] * 10, dtype=np.float32)

        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        r_hold = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w1.copy(),
        )

        reward_fn.reset()
        reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000],
            current_weights=w1,
        )
        w2_changed = w1 + 0.05
        r_change = reward_fn.compute(
            step_return=0.0,
            portfolio_values=[1_000_000, 1_000_000, 1_000_000],
            current_weights=w2_changed,
        )

        assert r_hold > r_change, "Holding steady should give better reward than rebalancing"

    def test_reset_clears_state(self, reward_fn):
        """reset() should clear all state for a new episode."""
        reward_fn.reset()
        reward_fn.compute(
            step_return=0.01,
            portfolio_values=[1_000_000, 1_010_000],
            current_weights=np.array([0.1] * 10, dtype=np.float32),
        )

        assert len(reward_fn._returns) > 0
        assert reward_fn._prev_weights is not None

        reward_fn.reset()
        assert len(reward_fn._returns) == 0
        assert reward_fn._prev_weights is None
        assert reward_fn._peak_value == 0.0

    def test_sharpe_component_after_warmup(self, reward_fn):
        """Reward computation should not produce NaN after warmup steps."""
        reward_fn.reset()

        for i in range(4):
            reward_fn.compute(
                step_return=0.001,
                portfolio_values=[1_000_000 + i * 1000] + [1_000_000 + (i + 1) * 1000],
            )

        r = reward_fn.compute(
            step_return=0.001,
            portfolio_values=list(range(1_000_000, 1_006_000, 1000)),
        )
        assert isinstance(r, float)
        assert not np.isnan(r)


class TestRewardAggregator:
    """Tests for RewardAggregator summary statistics."""

    def test_summary_keys(self):
        """Summary should contain expected keys."""
        from agent.reward import RewardAggregator
        agg = RewardAggregator()

        for i in range(100):
            agg.record(reward=0.01, step_return=0.001)

        summary = agg.summary()
        expected_keys = ["total_reward", "mean_reward", "cumulative_return_pct",
                         "annualised_sharpe", "max_drawdown_pct"]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_positive_returns_positive_sharpe(self):
        """Consistently positive (but varying) returns should give positive Sharpe."""
        from agent.reward import RewardAggregator
        np.random.seed(42)
        agg = RewardAggregator()

        for i in range(252):
            # Positive mean return with small noise so std > 0
            r = 0.001 + abs(np.random.randn() * 0.0005)
            agg.record(reward=r, step_return=r)

        summary = agg.summary()
        assert summary["annualised_sharpe"] > 0
        assert summary["cumulative_return_pct"] > 0

    def test_max_drawdown_non_negative(self):
        """Max drawdown should be non-negative."""
        from agent.reward import RewardAggregator
        agg = RewardAggregator()

        returns = [0.01, 0.01, -0.05, 0.01, -0.02, 0.01]
        for r in returns:
            agg.record(reward=r, step_return=r)

        summary = agg.summary()
        assert summary["max_drawdown_pct"] >= 0

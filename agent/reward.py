"""
agent/reward.py
───────────────
Risk-profile-parameterised reward functions for the RL trading agent.

The reward at each step is a weighted combination of:
  1. Step return          — immediate P&L signal
  2. Running Sharpe       — rewards consistent risk-adjusted performance
  3. Drawdown penalty     — discourages large peak-to-trough losses
  4. Turnover penalty     — HEAVY penalty on excessive rebalancing
  5. Holding bonus        — rewards the agent for staying in positions

v2 changes (from trade analysis):
  - Turnover penalty raised 10× — the #1 cause of losses was transaction
    costs from daily churning (65% of total losses in fold 0).
  - Added holding bonus — rewards the agent for keeping positions stable,
    directly counteracting the oscillation pattern (negative weight-change
    autocorrelation of -0.3 to -0.46).
  - Turnover penalty is now quadratic — small rebalances are cheap,
    large rebalances are punished exponentially.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class RiskAwareReward:
    """
    Computes a shaped, risk-profile-aware step reward.

    Parameters
    ----------
    config       : full config dict
    risk_profile : "conservative" | "balanced" | "aggressive"
    sharpe_window: rolling window (steps) over which Sharpe is estimated
    """

    def __init__(
        self,
        config: dict,
        risk_profile: str = "balanced",
        sharpe_window: int = 60,
        term: str = "medium",
    ):
        risk_cfg = config["risk_profiles"][risk_profile]
        self.sharpe_weight    = risk_cfg["sharpe_weight"]
        self.return_weight    = risk_cfg["return_weight"]
        self.drawdown_penalty = risk_cfg["drawdown_penalty"]
        self.reward_scaling   = config["env"].get("reward_scaling", 1.0)
        self.sharpe_window    = sharpe_window
        self.annualisation_factor = {
            "short": 252 * 7,
            "medium": 252,
            "long": 52,
        }.get(term, 252)

        # v2: turnover and holding parameters from config
        self.turnover_penalty_coef = config["env"].get("turnover_penalty", 100.0)
        self.holding_bonus_coef    = config["env"].get("holding_bonus", 0.02)

        # v3.1 Experiment D: Alpha-aware reward shaping
        # Bonus for acting in the direction of high-conviction alpha signals.
        # Only applied when alpha_signal is passed to compute().
        self.alpha_reward_bonus = config["env"].get("alpha_reward_bonus", 0.0)
        self.alpha_conviction_threshold = config["env"].get("alpha_conviction_threshold", 0.5)

        # Incremental state
        self._returns: List[float] = []
        self._peak_value: float    = 0.0
        self._prev_weights: np.ndarray | None = None

    def reset(self) -> None:
        """Called at the start of each episode."""
        self._returns      = []
        self._peak_value   = 0.0
        self._prev_weights = None

    def compute(
        self,
        step_return: float,
        portfolio_values: List[float],
        transaction_cost: float = 0.0,
        current_weights: np.ndarray | None = None,
        alpha_signal=None,
    ) -> float:
        """
        Compute the shaped reward for one environment step.

        Parameters
        ----------
        step_return       : fractional return for this step, e.g. 0.003
        portfolio_values  : full history of portfolio values (for drawdown)
        transaction_cost  : absolute transaction cost incurred this step
        current_weights   : current portfolio weight vector (for turnover calc)
        alpha_signal      : optional AlphaSignal — when provided and
                            alpha_reward_bonus > 0, gives a bonus for trades
                            aligned with high-conviction alpha predictions.

        Returns
        -------
        float : shaped scalar reward
        """
        self._returns.append(step_return)
        previous_weights = (
            self._prev_weights.copy() if self._prev_weights is not None else None
        )

        # ── Component 1: Raw return ───────────────────────────────────────────
        r_raw = step_return * self.return_weight

        # ── Component 2: Rolling Sharpe (annualised approximation) ───────────
        r_sharpe = 0.0
        if len(self._returns) >= 5:
            window_returns = np.array(self._returns[-self.sharpe_window:])
            mean_r  = np.mean(window_returns)
            std_r   = np.std(window_returns)
            if std_r > 1e-8:
                sharpe = mean_r / std_r * np.sqrt(self.annualisation_factor)
                r_sharpe = np.clip(sharpe, -3.0, 3.0) * self.sharpe_weight * 0.01

        # ── Component 3: Drawdown penalty ─────────────────────────────────────
        current_value = portfolio_values[-1]
        self._peak_value = max(self._peak_value, current_value)
        drawdown = 0.0
        if self._peak_value > 0:
            drawdown = (self._peak_value - current_value) / self._peak_value

        # Quadratic penalty on large drawdowns
        r_drawdown = -self.drawdown_penalty * (drawdown ** 2)

        # ── Component 4: Turnover penalty (HEAVY — v2) ───────────────────────
        # Two sub-components:
        #   a) Direct transaction cost penalty (proportional to cost incurred)
        #   b) Weight-change penalty (quadratic — punishes large rebalances hard)
        port_value = max(portfolio_values[-1], 1.0)

        # 4a: Direct cost penalty
        r_tc = -(transaction_cost / port_value) * self.turnover_penalty_coef

        # 4b: Weight-change penalty (quadratic)
        r_turnover = 0.0
        if current_weights is not None and previous_weights is not None:
            weight_change = np.abs(current_weights - previous_weights)
            # Quadratic: small changes (< 0.02) are nearly free,
            # large flips (0.2 → -0.2 = 0.4 change) are punished ~16× more
            turnover_sq = float(np.sum(weight_change ** 2))
            r_turnover = -self.turnover_penalty_coef * 0.5 * turnover_sq

        # ── Component 5: Holding bonus (v2) ──────────────────────────────────
        # Reward the agent for keeping positions stable. If the mean absolute
        # weight change across all stocks is < 0.01, the agent gets a bonus.
        # This directly counteracts the daily oscillation pattern.
        r_holding = 0.0
        if current_weights is not None and previous_weights is not None:
            mean_abs_change = float(np.mean(np.abs(current_weights - previous_weights)))
            if mean_abs_change < 0.02:
                # Full bonus for near-zero change, linearly decays to 0 at 0.02
                r_holding = self.holding_bonus_coef * (1.0 - mean_abs_change / 0.02)

        # ── Component 6: Alpha conviction bonus (v3.1 Experiment D) ──────────
        # Reward the agent for trading in the direction predicted by the alpha
        # model, but ONLY when the alpha model is confident (conviction above
        # threshold). This teaches the SAC agent to trust high-conviction signals.
        r_alpha = 0.0
        if (
            self.alpha_reward_bonus > 0
            and alpha_signal is not None
            and current_weights is not None
            and previous_weights is not None
        ):
            direction = getattr(alpha_signal, "direction", None)
            conviction = getattr(alpha_signal, "conviction", None)

            if direction is not None and conviction is not None:
                weight_delta = current_weights - previous_weights
                # For each stock: bonus if sign(weight_delta) == sign(direction)
                # AND conviction exceeds threshold
                aligned = np.sign(weight_delta) * np.sign(direction)  # +1 if aligned, -1 if opposed
                high_conviction = conviction >= self.alpha_conviction_threshold
                # Sum bonuses only for high-conviction aligned trades
                r_alpha = self.alpha_reward_bonus * float(
                    np.sum(aligned * conviction * high_conviction)
                )

        # Update state only after every component has compared with the same
        # pre-action portfolio.
        if current_weights is not None:
            self._prev_weights = current_weights.copy()

        # ── Aggregate ─────────────────────────────────────────────────────────
        reward = (r_raw + r_sharpe + r_drawdown + r_tc + r_turnover + r_holding + r_alpha) * self.reward_scaling

        logger.debug(
            "Reward — raw: %.4f | sharpe: %.4f | dd: %.4f | tc: %.4f | "
            "turnover: %.4f | holding: %.4f | alpha: %.4f | total: %.4f",
            r_raw, r_sharpe, r_drawdown, r_tc, r_turnover, r_holding, r_alpha, reward,
        )

        return float(reward)


class RewardAggregator:
    """
    Utility class for logging and analysing reward components over a full
    episode. Used in the backtesting runner and for diagnostics.
    """

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []

    def record(self, reward: float, step_return: float) -> None:
        self.episode_rewards.append(reward)
        self.episode_returns.append(step_return)

    def summary(self) -> dict:
        rewards  = np.array(self.episode_rewards)
        returns  = np.array(self.episode_returns)
        cum_ret  = float(np.prod(1 + returns) - 1)
        std_ret  = float(np.std(returns))
        sharpe   = float(np.mean(returns) / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0
        return {
            "total_reward":    float(np.sum(rewards)),
            "mean_reward":     float(np.mean(rewards)),
            "cumulative_return_pct": cum_ret * 100,
            "annualised_sharpe":     sharpe,
            "max_drawdown_pct":      self._max_drawdown(returns) * 100,
        }

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + returns)
        peak       = np.maximum.accumulate(cumulative)
        drawdown   = (peak - cumulative) / peak
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

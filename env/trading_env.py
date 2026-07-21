"""
env/trading_env.py
──────────────────
Gymnasium-compatible multi-stock NSE trading environment.

v2 changes (from trade post-mortem):
  - Action = DELTA weights, not absolute targets.
    The agent outputs weight *changes* ∈ [-max_delta, +max_delta] per stock.
    This naturally penalises large rebalances and forces deliberate trades.
  - Minimum holding period: once a position flips direction, it's locked
    for N steps. Prevents the daily oscillation pattern we observed.
  - Observation includes holding-period countdown so the agent knows
    which stocks are locked and can plan around it.

Design principles:
  - Observation = per-stock features + portfolio state + holding locks + sentiment
  - Action = delta-weights in [-max_delta, max_delta] per stock
  - Reward = risk-profile-parameterised signal (see agent/reward.py v2)
  - Transaction costs and slippage modelled explicitly
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from agent.reward import RiskAwareReward

logger = logging.getLogger(__name__)


class NSETradingEnv(gym.Env):
    """
    Multi-stock NSE paper trading environment with delta-weight actions.

    Parameters
    ----------
    data            : dict of {ticker: DataFrame} from NSEDataLoader
    feature_names   : ordered list of per-stock feature columns
    config          : full config dict
    risk_profile    : "conservative" | "balanced" | "aggressive"
    term            : "short" | "medium" | "long"
    sentiment_fn    : optional callable(step_date) → np.ndarray
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        feature_names: List[str],
        config: dict,
        risk_profile: str = "balanced",
        term: str = "medium",
        sentiment_fn=None,
        alpha_fn=None,
        temporal_fn=None,
        gnn_fn=None,
        regime_fn=None,
        debate_fn=None,
        terminate_on_stop: bool = True,
    ):
        super().__init__()

        configured_tickers = config.get("market", {}).get("tickers", [])
        ordered_tickers = [ticker for ticker in configured_tickers if ticker in data]
        ordered_tickers.extend(
            ticker for ticker in data if ticker not in ordered_tickers
        )
        self.tickers       = ordered_tickers
        self.data          = {ticker: data[ticker] for ticker in self.tickers}
        self.n_stocks      = len(self.tickers)
        self.feature_names = feature_names
        self.n_features    = len(feature_names)
        self.cfg           = config
        self.env_cfg       = config["env"]
        self.risk_cfg      = config["risk_profiles"][risk_profile]
        self.risk_profile  = risk_profile
        self.term          = term
        self.terminate_on_stop = terminate_on_stop
        self.sentiment_fn  = sentiment_fn
        self.sentiment_dim = config["llm"]["sentiment_dim"]

        # v3.1: Alpha signal function (LightGBM, or combined via meta-model)
        self.alpha_fn = alpha_fn
        if config.get("meta_model", {}).get("enabled", False):
            # Meta-model combines all channels into a single alpha signal
            self.alpha_dim = config.get("meta_model", {}).get("obs_dim", 20)
        else:
            self.alpha_dim = config.get("alpha", {}).get("obs_dim", 0)

        # v3.1 Phase 2: PatchTST temporal signal function
        # Dim is 0 when fn is None (e.g. when meta-model absorbs all channels).
        self.temporal_fn = temporal_fn
        self.temporal_dim = config.get("patchtst", {}).get("obs_dim", 0) if temporal_fn is not None else 0

        # v3.1 Phase 3: GNN cross-stock signal
        self.gnn_fn = gnn_fn
        self.gnn_dim = (
            config.get("gnn", {}).get("obs_dim", 0)
            if gnn_fn is not None and config.get("gnn", {}).get("enabled", False)
            else 0
        )

        # v3.1 Phase 3: VAE regime embedding
        self.regime_fn = regime_fn
        self.regime_dim = (
            config.get("vae_regime", {}).get("obs_dim", 0)
            if regime_fn is not None and config.get("vae_regime", {}).get("enabled", False)
            else 0
        )

        # Phase 4: Debate analyst signal
        self.debate_fn = debate_fn
        self.debate_dim = (
            config.get("llm", {}).get("debate", {}).get("obs_dim", 0)
            if debate_fn is not None and config.get("llm", {}).get("debate", {}).get("enabled", False)
            else 0
        )

        # v2: Delta-weight action space parameters
        self.max_delta = self.env_cfg.get("max_weight_delta", 0.05)
        self.min_hold_steps = self.env_cfg.get("min_hold_steps", 5)
        self.execution_lag_bars = max(
            int(self.env_cfg.get("execution_lag_bars", 1)),
            1,
        )

        # Align all tickers to common date index
        self._dates = self._get_common_dates()
        self._n_steps = len(self._dates)
        if self._n_steps < self.execution_lag_bars + 2:
            raise ValueError(
                "Trading environment needs at least 3 common dates for "
                "lagged observation, execution, and valuation."
            )
        # Causal clock: features/signals from t-1 decide an execution at t.
        self._current_step = self.execution_lag_bars

        # Portfolio state
        self.initial_capital = self.env_cfg["initial_capital"]
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_stocks)        # shares held
        self.weights   = np.zeros(self.n_stocks)        # current weights
        self.prev_value = self.initial_capital
        self._stopped_out = False

        # v2: Holding period lock — countdown per stock.
        # When > 0, the stock's weight direction cannot be flipped.
        self._hold_lock = np.zeros(self.n_stocks, dtype=np.int32)

        # Transaction costs
        self.tc_pct       = self.env_cfg["transaction_cost_pct"]
        self.slippage_pct = self.env_cfg["slippage_pct"]

        # Reward function
        self.reward_fn = RiskAwareReward(config, risk_profile, term=term)

        # Track portfolio history for Sharpe / drawdown calculation
        self._value_history: List[float] = [self.initial_capital]
        self._return_history: List[float] = []

        # First-miss logging flags — log once per signal type when it falls
        # back to zeros, so cold-start failures aren't silently swallowed.
        self._signal_miss_logged = {
            "alpha": False, "temporal": False, "gnn": False,
            "regime": False, "debate": False, "sentiment": False,
        }
        self._channel_enabled = {
            "alpha": self.alpha_fn is not None and self.alpha_dim > 0,
            "temporal": self.temporal_fn is not None and self.temporal_dim > 0,
            "gnn": self.gnn_fn is not None and self.gnn_dim > 0,
            "regime": self.regime_fn is not None and self.regime_dim > 0,
            "debate": self.debate_fn is not None and self.debate_dim > 0,
        }
        self._channel_health = {
            name: {"hits": 0, "misses": 0}
            for name in self._channel_enabled
        }
        self.fail_on_dead_channel = self.env_cfg.get("fail_on_dead_channel")
        self._dead_channel_probe_steps = 200
        self._dead_channel_checked = False
        # Market rows and fitted-model outputs are fold-static. Lazy caching
        # turns millions of repeated pandas/model calls into O(1) array lookup
        # after the first episode while preserving stochastic sentiment
        # augmentation at training time.
        self._market_obs_cache: Dict[pd.Timestamp, np.ndarray] = {}
        self._signal_cache: Dict[str, Dict[pd.Timestamp, np.ndarray]] = {
            "alpha": {},
            "temporal": {},
            "gnn": {},
            "regime": {},
            "debate": {},
        }
        self._signal_status: Dict[str, Dict[pd.Timestamp, bool]] = {
            channel: {} for channel in self._signal_cache
        }

        # ── Spaces ────────────────────────────────────────────────────────────
        # Observation: [market | portfolio | locks | sentiment | alpha | temporal]
        obs_dim = (
            self.n_stocks * self.n_features          # market features
            + self.n_stocks                          # current weights
            + 2                                      # portfolio value, cash ratio
            + self.n_stocks                          # holding lock countdown (normalised)
            + self.n_stocks * self.sentiment_dim     # LLM sentiment per ticker
            + self.alpha_dim                         # v3.1 P1: LightGBM alpha signal
            + self.temporal_dim                      # v3.1 P2: PatchTST temporal signal
            + self.gnn_dim                           # v3.1 P3: GNN cross-stock signal
            + self.regime_dim                        # v3.1 P3: VAE regime embedding
            + self.debate_dim                        # Phase 4: debate analyst signal
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # v2: Actions are DELTA weights, not absolute targets.
        # Range [-1, 1] is rescaled to [-max_delta, +max_delta] internally.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Start at the second bar. The first observation uses bar 0 information
        # and the first action executes at bar 1, avoiding a same-close fill.
        self._current_step = self.execution_lag_bars
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions  = np.zeros(self.n_stocks)
        self.weights    = np.zeros(self.n_stocks)
        self.prev_value = self.initial_capital
        self._stopped_out = False
        self._hold_lock = np.zeros(self.n_stocks, dtype=np.int32)
        self._value_history  = [self.initial_capital]
        self._return_history = []
        self.reward_fn.reset()
        self._channel_health = {
            name: {"hits": 0, "misses": 0}
            for name in self._channel_enabled
        }
        self._dead_channel_checked = False

        obs = self._get_observation()

        # One-time sanity check: catch obs_dim mismatches at the first reset
        # instead of surfacing them deep inside SAC.predict() much later.
        # The classic failure mode is a channel fn (alpha/temporal/gnn/regime/
        # debate) being None on one env but populated on another, or returning
        # a vector whose length doesn't match its config.obs_dim.
        if not getattr(self, "_obs_shape_validated", False):
            declared = self.observation_space.shape[0]
            actual   = obs.shape[0]
            if actual != declared:
                raise RuntimeError(
                    f"NSETradingEnv obs shape mismatch: declared "
                    f"observation_space=({declared},) but _get_observation() "
                    f"returned ({actual},). Likely cause: a channel fn "
                    f"(alpha/temporal/gnn/regime/debate) is None vs non-None "
                    f"differently from what sized obs_dim in __init__, "
                    f"OR a signal's to_obs_vector() returned an unexpected "
                    f"length. Component dims: market={self.n_stocks * self.n_features}, "
                    f"weights={self.n_stocks}, port=2, lock={self.n_stocks}, "
                    f"sentiment={self.n_stocks * self.sentiment_dim}, "
                    f"alpha={self.alpha_dim}, temporal={self.temporal_dim}, "
                    f"gnn={self.gnn_dim}, regime={self.regime_dim}, "
                    f"debate={self.debate_dim}."
                )
            self._obs_shape_validated = True

        return obs, self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step.

        action : array of shape (n_stocks,), values in [-1, 1]
                 Rescaled to [-max_delta, +max_delta] and ADDED to current weights.
        """
        assert not self._is_done(), "Episode is done. Call reset()."

        # v3.2 P0: NaN guard on actions
        if np.any(np.isnan(action)):
            logger.warning(
                "NaN detected in action at step %d — replacing with zeros.",
                self._current_step,
            )
            action = np.nan_to_num(action, nan=0.0)

        # Clamp to [-1, 1] (SAC should output in range but be safe)
        action = np.clip(action, -1.0, 1.0)
        if self._stopped_out:
            # Evaluation keeps a stopped portfolio in cash for the remainder
            # of the fold instead of truncating and biasing annualised metrics.
            action = np.zeros_like(action)

        max_w = self.risk_cfg["max_position_pct"]

        # ── v2: Convert action to delta and apply to current weights ─────────
        # Scale from [-1, 1] → [-max_delta, +max_delta]
        delta = action * self.max_delta

        # Apply holding period lock: if a stock is locked and the delta
        # would flip its direction, zero out that delta.
        for i in range(self.n_stocks):
            if self._hold_lock[i] > 0:
                current_sign = np.sign(self.weights[i])
                proposed = self.weights[i] + delta[i]
                # Block if it would flip direction (but allow reducing position)
                if current_sign != 0 and np.sign(proposed) != current_sign and np.sign(proposed) != 0:
                    delta[i] = 0.0

        # Apply delta to get new target weights
        target_weights = self.weights + delta

        # Cash-equity mode is long-only unless the experiment explicitly models
        # a shortable instrument, borrow costs, and funding.
        lower_bound = -max_w if self.env_cfg.get("allow_short", False) else 0.0
        target_weights = np.clip(target_weights, lower_bound, max_w)

        # Enforce leverage constraint
        leverage = self.risk_cfg["leverage"]
        total_gross = np.sum(np.abs(target_weights))
        if total_gross > leverage:
            target_weights = target_weights / total_gross * leverage

        # ── v2: Update holding lock countdowns ───────────────────────────────
        # Decrement all locks
        self._hold_lock = np.maximum(self._hold_lock - 1, 0)

        # If a stock's weight FLIPPED direction this step, set its lock
        for i in range(self.n_stocks):
            old_sign = np.sign(self.weights[i])
            new_sign = np.sign(target_weights[i])
            if old_sign != 0 and new_sign != 0 and old_sign != new_sign:
                self._hold_lock[i] = self.min_hold_steps

        # ── Rebalance portfolio → compute transaction costs ──────────────────
        execution_step = self._current_step
        signal_step = max(
            execution_step - self.execution_lag_bars,
            0,
        )
        execution_date = self._dates[execution_step]
        signal_date = self._dates[signal_step]
        prices = self._get_prices(execution_step)
        tc     = self._rebalance(target_weights, prices)
        reward_weights = target_weights.copy()

        # Update weights
        self.weights = target_weights

        # ── Advance to next step ─────────────────────────────────────────────
        self._current_step += 1
        done = self._is_done()

        # Mark-to-market portfolio value after price move
        next_prices = self._get_prices(self._current_step)
        self.portfolio_value = (
            self.cash + np.sum(self.positions * next_prices)
        )

        # Actual weights drift with prices; carrying stale target weights makes
        # later deltas, turnover, and logs economically incorrect.
        if self.portfolio_value > 1e-8:
            self.weights = (
                self.positions * next_prices / self.portfolio_value
            ).astype(np.float64)

        stop_triggered = (not self._stopped_out) and self._check_stop_loss()
        if stop_triggered:
            logger.warning(
                "Stop-loss triggered at step %d | drawdown=%.1f%% > threshold=%.1f%% | "
                "Liquidating portfolio (value=₹%.0f).",
                self._current_step,
                self._compute_drawdown() * 100,
                self.risk_cfg["stop_loss_pct"] * 100,
                self.portfolio_value,
            )
            # Liquidation is a real trade: include adverse slippage and fees.
            tc += self._rebalance(np.zeros(self.n_stocks), next_prices)
            self.portfolio_value = self.cash
            self.weights = np.zeros(self.n_stocks)
            reward_weights = self.weights.copy()
            self._stopped_out = True

        # Compute return after all execution and liquidation costs.
        self._value_history.append(self.portfolio_value)
        step_return = (self.portfolio_value - self.prev_value) / max(self.prev_value, 1.0)
        self._return_history.append(step_return)
        self.prev_value = self.portfolio_value

        # ── Compute reward (v2 — with weights for turnover/holding calc) ─────
        # v3.1 Experiment D: pass alpha signal to reward function for
        # conviction-aligned bonus (only active when alpha_reward_bonus > 0)
        alpha_sig_for_reward = None
        if self.alpha_fn is not None and self.env_cfg.get("alpha_reward_bonus", 0) > 0:
            try:
                alpha_sig_for_reward = self.alpha_fn(signal_date)
            except Exception:
                pass

        reward = self.reward_fn.compute(
            step_return=step_return,
            portfolio_values=self._value_history,
            transaction_cost=tc,
            current_weights=reward_weights,
            alpha_signal=alpha_sig_for_reward,
        )

        # Terminal penalty on stop-loss truncation. Without it, the agent learns
        # to deliberately trigger the stop-loss to cap downside risk via early
        # episode termination — observed as eval reward regression and episode
        # length collapsing to a fixed value across deterministic evals.
        if stop_triggered:
            reward += -float(self.risk_cfg.get("stop_loss_terminal_penalty", 5.0))

        obs      = self._get_observation()
        info     = self._get_info(
            tc=tc,
            stop_triggered=stop_triggered,
            signal_date=signal_date,
            execution_date=execution_date,
        )
        truncated = stop_triggered and self.terminate_on_stop
        if done:
            self._log_channel_health()

        return obs, reward, done, truncated, info

    def render(self) -> None:
        """Print a one-line portfolio summary to stdout."""
        pnl = self.portfolio_value - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100
        date = self._dates[min(self._current_step, len(self._dates) - 1)]
        print(
            f"[{self.risk_profile.upper():12s}] "
            f"Step {self._current_step:4d}/{self._n_steps} | "
            f"Date: {date} | "
            f"Portfolio: ₹{self.portfolio_value:,.0f} | "
            f"P&L: {pnl_pct:+.2f}%"
        )

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        # Features and exogenous signals are lagged one bar relative to the
        # execution clock. An observation built while current_step=t can only
        # contain information available at t-1.
        step = max(
            min(
                self._current_step - self.execution_lag_bars,
                self._n_steps - 1,
            ),
            0,
        )
        date = self._dates[step]

        # Per-stock market features
        market_obs = self._market_obs_cache.get(date)
        if market_obs is None:
            market_features = []
            for ticker in self.tickers:
                df  = self.data[ticker]
                row = (
                    df.loc[date, self.feature_names].values
                    if date in df.index
                    else np.zeros(self.n_features)
                )
                market_features.append(row)
            market_obs = np.concatenate(market_features).astype(np.float32)
            self._market_obs_cache[date] = market_obs

        # Portfolio state
        port_value_norm = np.log(self.portfolio_value / self.initial_capital)
        cash_ratio      = self.cash / max(self.portfolio_value, 1.0)
        portfolio_obs   = np.append(self.weights, [port_value_norm, cash_ratio]).astype(np.float32)

        # v2: Holding lock countdown (normalised to [0, 1])
        lock_obs = (self._hold_lock / max(self.min_hold_steps, 1)).astype(np.float32)

        # LLM sentiment vector
        if self.sentiment_fn is not None:
            try:
                sentiment = self.sentiment_fn(date).astype(np.float32)
            except Exception as e:
                if not self._signal_miss_logged["sentiment"]:
                    logger.warning("Sentiment signal failed on first call (date=%s): %s — using zeros.", date, e)
                    self._signal_miss_logged["sentiment"] = True
                sentiment = np.zeros(self.n_stocks * self.sentiment_dim, dtype=np.float32)
        else:
            sentiment = np.zeros(self.n_stocks * self.sentiment_dim, dtype=np.float32)

        # Fitted-model channels are deterministic within a fold and cached.
        alpha_obs = self._get_cached_signal(
            "alpha", date, self.alpha_fn, self.alpha_dim
        )
        self._record_channel_health("alpha", alpha_obs, date)

        temporal_obs = self._get_cached_signal(
            "temporal", date, self.temporal_fn, self.temporal_dim
        )
        self._record_channel_health("temporal", temporal_obs, date)

        gnn_obs = self._get_cached_signal(
            "gnn", date, self.gnn_fn, self.gnn_dim
        )
        self._record_channel_health("gnn", gnn_obs, date)

        regime_obs = self._get_cached_signal(
            "regime", date, self.regime_fn, self.regime_dim
        )
        self._record_channel_health("regime", regime_obs, date)

        debate_obs = self._get_cached_signal(
            "debate", date, self.debate_fn, self.debate_dim
        )
        self._record_channel_health("debate", debate_obs, date)

        obs = np.concatenate([market_obs, portfolio_obs, lock_obs, sentiment, alpha_obs, temporal_obs, gnn_obs, regime_obs, debate_obs])
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

        # v3.2 P0: Periodic observation diagnostics
        if self._current_step % 50 == 0:
            zeros_pct = float(np.mean(obs == 0) * 100)
            clipped_pct = float(np.mean(np.abs(obs) >= 9.9) * 100)
            logger.debug(
                "Obs diagnostics step=%d: mean=%.3f std=%.3f min=%.2f max=%.2f "
                "zeros=%.1f%% clipped=%.1f%%",
                self._current_step, float(np.mean(obs)), float(np.std(obs)),
                float(np.min(obs)), float(np.max(obs)), zeros_pct, clipped_pct,
            )
            if zeros_pct > 80:
                logger.warning(
                    "Step %d: %.0f%% of observation is zeros — possible signal failure.",
                    self._current_step, zeros_pct,
                )
            if clipped_pct > 20:
                logger.warning(
                    "Step %d: %.0f%% of observation is near clip boundary — "
                    "possible scaling issue.",
                    self._current_step, clipped_pct,
                )
        self._check_dead_channels()

        return obs

    def _get_cached_signal(
        self,
        channel: str,
        date,
        signal_fn,
        expected_dim: int,
    ) -> np.ndarray:
        if signal_fn is None or expected_dim <= 0:
            return np.zeros(expected_dim, dtype=np.float32)
        cached = self._signal_cache[channel].get(date)
        if cached is not None:
            return cached
        try:
            vector = signal_fn(date).to_obs_vector().astype(np.float32)
            if vector.shape != (expected_dim,):
                raise ValueError(
                    f"{channel} returned {vector.shape}, expected "
                    f"({expected_dim},)"
                )
            self._signal_status[channel][date] = True
        except Exception as exc:
            if not self._signal_miss_logged[channel]:
                logger.warning(
                    "%s signal failed on first call (date=%s): %s — using zeros.",
                    channel.capitalize(),
                    date,
                    exc,
                )
                self._signal_miss_logged[channel] = True
            vector = np.zeros(expected_dim, dtype=np.float32)
            self._signal_status[channel][date] = False
        self._signal_cache[channel][date] = vector
        return vector

    def _record_channel_health(
        self,
        channel: str,
        obs_vec: np.ndarray,
        date=None,
    ) -> None:
        """Track per-channel availability as non-zero vs zero fallback."""
        if not self._channel_enabled.get(channel, False):
            return
        available = (
            self._signal_status[channel].get(date)
            if date is not None
            else None
        )
        if available is True:
            self._channel_health[channel]["hits"] += 1
        elif available is False:
            self._channel_health[channel]["misses"] += 1
        elif np.any(np.abs(obs_vec) > 1e-12):
            self._channel_health[channel]["hits"] += 1
        else:
            self._channel_health[channel]["misses"] += 1

    def _channel_hit_rate(self, channel: str) -> float:
        counts = self._channel_health[channel]
        total = counts["hits"] + counts["misses"]
        return counts["hits"] / total if total > 0 else 0.0

    def _log_channel_health(self) -> None:
        parts = []
        for channel in ("alpha", "temporal", "gnn", "regime", "debate"):
            if not self._channel_enabled.get(channel, False):
                continue
            rate = self._channel_hit_rate(channel)
            parts.append(f"{channel}={rate * 100:.0f}%")
        if parts:
            logger.info("Channel health: %s", " ".join(parts))

    def _check_dead_channels(self) -> None:
        """Optionally fail fast when channels are mostly dead early in episode."""
        threshold = self.fail_on_dead_channel
        if threshold is None or self._dead_channel_checked:
            return
        if self._current_step < self._dead_channel_probe_steps:
            return

        self._dead_channel_checked = True
        dead = []
        for channel in ("alpha", "temporal", "gnn", "regime", "debate"):
            if not self._channel_enabled.get(channel, False):
                continue
            counts = self._channel_health[channel]
            total = counts["hits"] + counts["misses"]
            if total < self._dead_channel_probe_steps:
                continue
            rate = self._channel_hit_rate(channel)
            if rate < float(threshold):
                dead.append((channel, rate))

        if dead:
            dead_txt = ", ".join(f"{name}={rate * 100:.0f}%" for name, rate in dead)
            raise RuntimeError(
                f"Dead channel guard triggered at step {self._current_step}: "
                f"{dead_txt} below threshold={float(threshold):.2f}."
            )

    # ── Portfolio mechanics ───────────────────────────────────────────────────

    def _rebalance(
        self, target_weights: np.ndarray, prices: np.ndarray
    ) -> float:
        """
        Trade from current positions to target weights.
        Returns total transaction cost incurred.
        """
        target_values = target_weights * self.portfolio_value
        target_shares = target_values / np.maximum(prices, 1e-8)
        delta_shares  = target_shares - self.positions

        # Apply slippage: adverse fill price for each trade
        fill_prices = prices * (1 + np.sign(delta_shares) * self.slippage_pct)

        # Transaction cost on notional traded
        notional = np.abs(delta_shares * fill_prices)
        tc       = float(np.sum(notional * self.tc_pct))

        # Update cash and positions
        self.cash      -= float(np.sum(delta_shares * fill_prices)) + tc
        self.positions  = target_shares
        return tc

    def _check_stop_loss(self) -> bool:
        """Return True if portfolio has fallen below the stop-loss threshold."""
        return bool(self._compute_drawdown() >= self.risk_cfg["stop_loss_pct"])

    def _compute_drawdown(self) -> float:
        """Compute current drawdown from peak."""
        peak = max(self._value_history)
        if peak <= 0:
            return 0.0
        return (peak - self.portfolio_value) / peak

    def _get_prices(self, step: int) -> np.ndarray:
        """Return close prices for all tickers at the given step index."""
        step = min(step, self._n_steps - 1)
        date = self._dates[step]
        prices = []
        for ticker in self.tickers:
            df = self.data[ticker]
            p  = float(df.loc[date, "Close"]) if date in df.index else 1.0
            prices.append(p)
        return np.array(prices, dtype=np.float32)

    def _is_done(self) -> bool:
        return self._current_step >= self._n_steps - 1

    def _get_info(
        self,
        tc: float = 0.0,
        stop_triggered: bool = False,
        signal_date=None,
        execution_date=None,
    ) -> dict:
        """Return diagnostic info dict for logging and evaluation."""
        valuation_date = self._dates[min(self._current_step, self._n_steps - 1)]
        if signal_date is None:
            signal_date = self._dates[
                max(
                    min(
                        self._current_step - self.execution_lag_bars,
                        self._n_steps - 1,
                    ),
                    0,
                )
            ]
        if execution_date is None:
            execution_date = valuation_date
        return {
            "step":            self._current_step,
            "date":            str(valuation_date),
            "signal_date":     str(signal_date),
            "execution_date":  str(execution_date),
            "portfolio_value": self.portfolio_value,
            "cash":            self.cash,
            "weights":         self.weights.tolist(),
            "positions":       self.positions.tolist(),
            "transaction_cost": tc,
            "stop_triggered":  stop_triggered,
            "risk_profile":    self.risk_profile,
            "term":            self.term,
            "pnl_pct": (self.portfolio_value - self.initial_capital) / self.initial_capital * 100,
        }

    def _get_common_dates(self) -> pd.Index:
        """
        Find the intersection of all ticker date indices, keeping only
        dates where every ticker has non-NaN values for ALL features.

        This eliminates the indicator warm-up period (first ~50-60 rows)
        where rolling indicators like MACD, Bollinger Bands, and 50-day
        EMA are NaN. Without this, the observation vector is 96% zeros
        during warm-up, causing the SAC policy to collapse immediately.
        """
        common = None
        for ticker, df in self.data.items():
            # Only keep dates where all features are valid (non-NaN)
            feat_cols = [c for c in self.feature_names if c in df.columns]
            if feat_cols:
                valid = df.dropna(subset=feat_cols).index
            else:
                valid = df.index
            idx = set(valid)
            common = idx if common is None else common & idx

        result = pd.DatetimeIndex(sorted(common))

        # Log how many dates were dropped due to NaN features
        all_dates = None
        for df in self.data.values():
            idx = set(df.index)
            all_dates = idx if all_dates is None else all_dates & idx
        n_dropped = len(all_dates) - len(result)
        if n_dropped > 0:
            logger.info(
                "Dropped %d dates with NaN features (indicator warm-up). "
                "Trading on %d clean dates.",
                n_dropped, len(result),
            )

        return result

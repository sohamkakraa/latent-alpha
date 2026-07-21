"""
strategy/multi_horizon.py
═════════════════════════
Multi-horizon portfolio orchestrator.

Runs short/medium/long horizon agents in parallel and combines
their portfolio weight recommendations into a single allocation.

Combination strategies:
  1. Equal weight (default) — simple average
  2. Risk-parity — weight inversely by recent volatility
  3. Sharpe-weighted — weight by rolling Sharpe ratio

The orchestrator handles:
  - Different rebalancing frequencies (daily / weekly / monthly)
  - Capital allocation across horizons
  - Conflict resolution when horizons disagree
"""

from __future__ import annotations

import logging
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Named tuple for each active horizon's state
HorizonAgent = namedtuple(
    "HorizonAgent",
    ["term", "agent", "env", "last_weights", "last_rebalance_date"],
)


class MultiHorizonOrchestrator:
    """
    Combines short/medium/long horizon SAC agents into a single portfolio.

    Parameters
    ----------
    config       : full config dict
    risk_profile : "conservative" | "balanced" | "aggressive"
    horizons     : list of active terms, e.g. ["medium", "long"]
    """

    REBALANCE_FREQ = {
        "short":  1,   # Every step (daily bars)
        "medium": 5,   # Every 5 steps (~weekly)
        "long":   21,  # Every 21 steps (~monthly)
    }

    def __init__(
        self,
        config: dict,
        risk_profile: str = "balanced",
        horizons: Optional[List[str]] = None,
    ):
        self.config       = config
        self.risk_profile = risk_profile
        mh_cfg            = config.get("multi_horizon", {})
        self.horizons     = horizons or mh_cfg.get("horizons", ["medium"])
        self.combination  = mh_cfg.get("combination", "equal_weight")
        self.capital_split = mh_cfg.get("capital_split", {h: 1.0 / len(self.horizons) for h in self.horizons})
        self._agents: Dict[str, HorizonAgent] = {}
        self._step_count = 0

        logger.info(
            "MultiHorizonOrchestrator | horizons=%s | combination=%s",
            self.horizons, self.combination,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_start: str,
        train_end: str,
        sentiment_fn=None,
        total_timesteps: Optional[int] = None,
    ) -> None:
        """Train all horizon agents sequentially."""
        from strategy.term_selector import TermSelector
        from agent.sac_agent import LatentAlphaSACAgent

        selector = TermSelector(self.config)

        for term in self.horizons:
            logger.info("Training %s-horizon agent...", term)
            train_env, eval_env = selector.build_env(
                risk_profile=self.risk_profile,
                term=term,
                train_start=train_start,
                train_end=train_end,
                sentiment_fn=sentiment_fn,
            )
            agent = LatentAlphaSACAgent(self.config, self.risk_profile, term)
            agent.train(
                train_env,
                eval_env=eval_env,
                total_timesteps=total_timesteps,
                resume=False,
            )
            n_stocks = len(self.config["market"]["tickers"])
            self._agents[term] = HorizonAgent(
                term=term,
                agent=agent,
                env=eval_env,
                last_weights=np.zeros(n_stocks, dtype=np.float32),
                last_rebalance_date=None,
            )
            logger.info("Trained %s-horizon agent.", term)

    # ── Inference ─────────────────────────────────────────────────────────────

    def step(self, obs_by_term: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine weights from all active horizons.

        Parameters
        ----------
        obs_by_term : dict of {term: observation_array}

        Returns
        -------
        Combined portfolio weights (n_stocks,)
        """
        self._step_count += 1
        horizon_weights: Dict[str, np.ndarray] = {}

        for term, ha in self._agents.items():
            # Only rebalance at the term's natural frequency
            if self._step_count % self.REBALANCE_FREQ.get(term, 1) == 0:
                obs = obs_by_term.get(term)
                if obs is not None:
                    try:
                        action, _ = ha.agent.predict(obs, deterministic=True)
                        max_delta = self.config["env"].get("max_weight_delta", 0.05)
                        new_weights = ha.last_weights + action * max_delta
                        new_weights = np.clip(new_weights, -0.35, 0.35)
                        # Update in-place (namedtuple is immutable — replace)
                        self._agents[term] = ha._replace(last_weights=new_weights)
                    except Exception as e:
                        logger.warning("Horizon %s predict failed: %s", term, e)

            horizon_weights[term] = self._agents[term].last_weights

        if not horizon_weights:
            n_stocks = len(self.config["market"]["tickers"])
            return np.zeros(n_stocks, dtype=np.float32)

        if self.combination == "equal_weight":
            return self._equal_weight(horizon_weights)
        elif self.combination == "risk_parity":
            vols = {t: float(np.std(w)) for t, w in horizon_weights.items()}
            return self._risk_parity(horizon_weights, vols)
        else:
            return self._equal_weight(horizon_weights)

    # ── Combination strategies ────────────────────────────────────────────────

    @staticmethod
    def _equal_weight(weights_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average across horizons."""
        if not weights_dict:
            return np.array([], dtype=np.float32)
        stacked = np.stack(list(weights_dict.values()), axis=0)
        combined = stacked.mean(axis=0).astype(np.float32)
        # Clip total exposure to 1.5 (1.5× leverage max)
        total = np.sum(np.abs(combined))
        if total > 1.5:
            combined = combined / total * 1.5
        return combined

    @staticmethod
    def _risk_parity(
        weights_dict: Dict[str, np.ndarray],
        volatilities: Dict[str, float],
    ) -> np.ndarray:
        """Weight inversely by recent volatility."""
        if not weights_dict:
            return np.array([], dtype=np.float32)

        inv_vols = {}
        for term, vol in volatilities.items():
            inv_vols[term] = 1.0 / max(vol, 1e-6)

        total_inv = sum(inv_vols.values())
        combined  = np.zeros_like(list(weights_dict.values())[0], dtype=np.float32)
        for term, w in weights_dict.items():
            combined += (inv_vols[term] / total_inv) * np.array(w, dtype=np.float32)

        total = np.sum(np.abs(combined))
        if total > 1.5:
            combined = combined / total * 1.5
        return combined

    @staticmethod
    def _sharpe_weighted(
        weights_dict: Dict[str, np.ndarray],
        sharpes: Dict[str, float],
    ) -> np.ndarray:
        """Weight by rolling Sharpe ratio (positive Sharpe only)."""
        if not weights_dict:
            return np.array([], dtype=np.float32)

        pos_sharpes = {t: max(s, 0.0) for t, s in sharpes.items()}
        total_sharpe = sum(pos_sharpes.values())
        if total_sharpe <= 0:
            return MultiHorizonOrchestrator._equal_weight(weights_dict)

        combined = np.zeros_like(list(weights_dict.values())[0], dtype=np.float32)
        for term, w in weights_dict.items():
            combined += (pos_sharpes[term] / total_sharpe) * np.array(w, dtype=np.float32)

        total = np.sum(np.abs(combined))
        if total > 1.5:
            combined = combined / total * 1.5
        return combined

    # ── Utilities ─────────────────────────────────────────────────────────────

    def combine_weights(
        self,
        horizon_weights: Dict[str, np.ndarray],
        strategy: str = "equal_weight",
    ) -> np.ndarray:
        """Public interface for combining pre-computed horizon weights."""
        if strategy == "equal_weight":
            return self._equal_weight(horizon_weights)
        elif strategy == "risk_parity":
            vols = {t: float(np.std(w)) for t, w in horizon_weights.items()}
            return self._risk_parity(horizon_weights, vols)
        else:
            return self._equal_weight(horizon_weights)

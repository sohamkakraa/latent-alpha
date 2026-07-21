"""
alpha/meta_model.py
═══════════════════
Stacking meta-model scaffold for combining multiple alpha sources.

Current implementation: Ridge regression combining LightGBM predictions.
Future: Will stack LightGBM + PatchTST + GNN + VAE regime signals
into a unified alpha signal via learned combination weights.

Architecture (v3.1 Phase 1 — scaffold):
  Input channels (current):
    - LightGBM cross-sectional alpha (direction + conviction)

  Input channels (future phases):
    - PatchTST temporal encoding (Phase 2)
    - GNN cross-stock dependency signal (Phase 3)
    - VAE regime embedding (Phase 3)
    - LLM sentiment aggregation (Phase 4)

  Output:
    - Combined AlphaSignal per stock

Design decisions:
  - Ridge regression chosen for interpretability and resistance to overfitting
    with small number of input channels.
  - Will migrate to a small neural net when we have 4+ input channels.
  - Channels are registered dynamically — adding a new model source
    requires only registering it, not changing the meta-model code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChannelPrediction:
    """
    A single prediction channel (one alpha source).

    Attributes
    ----------
    name       : Channel identifier (e.g., "lightgbm", "patchtst", "gnn")
    direction  : Predicted direction per stock (n_stocks,)
    conviction : Model confidence per stock (n_stocks,) in [0, 1]
    weight     : Channel weight in ensemble (learned by meta-model)
    """
    name: str
    direction: np.ndarray
    conviction: np.ndarray
    weight: float = 1.0


class RidgeMetaModel:
    """
    Ridge regression meta-model for stacking alpha signals.

    Usage
    -----
    meta = RidgeMetaModel(config)
    meta.register_channel("lightgbm")
    meta.register_channel("patchtst")   # future
    meta.train(channel_preds, actuals)
    combined = meta.combine(channel_preds)
    """

    def __init__(self, config: dict = None, n_stocks: int = None, ridge_alpha: float = None):
        # Support both config-dict style and direct kwargs style
        self.cfg = config or {}
        if ridge_alpha is not None:
            self.ridge_alpha = ridge_alpha
        else:
            alpha_cfg = self.cfg.get("alpha", {})
            meta_cfg = self.cfg.get("meta_model", {})
            self.ridge_alpha = meta_cfg.get("ridge_alpha", alpha_cfg.get("meta_ridge_alpha", 1.0))
        self.n_stocks = n_stocks

        # Registered channels
        self._channels: Dict[str, float] = {}  # name → learned weight
        self._is_trained = False

        # Ridge model
        self._ridge = None
        self._intercept = 0.0

    # ── Channel management ───────────────────────────────────────────────────

    def register_channel(self, name: str, initial_weight: float = 1.0):
        """Register a new prediction channel."""
        self._channels[name] = initial_weight
        logger.info("Registered meta-model channel: %s (weight=%.2f)", name, initial_weight)

    def get_channels(self) -> List[str]:
        """Return list of registered channel names."""
        return list(self._channels.keys())

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        channel_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the meta-model to combine channel predictions.

        Parameters
        ----------
        channel_predictions : dict of {channel_name: predictions array (n_samples,)}
        actuals             : actual forward returns (n_samples,)

        Returns
        -------
        Dict with learned weights and R² score
        """
        channels = sorted(channel_predictions.keys())
        if len(channels) == 0:
            raise ValueError("No channel predictions provided.")

        # Build feature matrix: each column is a channel's predictions
        X = np.column_stack([channel_predictions[c] for c in channels])
        y = actuals

        # Remove NaN rows
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid]
        y = y[valid]

        if len(X) < 10:
            logger.warning("Too few samples (%d) for meta-model training.", len(X))
            # Equal weights fallback
            for ch in channels:
                self._channels[ch] = 1.0 / len(channels)
            self._is_trained = True
            return {"r2": 0.0, "weights": dict(self._channels)}

        # Ridge regression (closed-form solution)
        n_features = X.shape[1]
        I = np.eye(n_features)
        XtX = X.T @ X
        Xty = X.T @ y

        # Add L2 regularisation
        weights = np.linalg.solve(
            XtX + self.ridge_alpha * I, Xty
        )

        # Store learned weights
        for ch, w in zip(channels, weights):
            self._channels[ch] = float(w)

        # Compute R²
        y_pred = X @ weights
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._is_trained = True

        logger.info(
            "Meta-model trained: R²=%.4f, weights=%s",
            r2, {ch: f"{w:.3f}" for ch, w in zip(channels, weights)},
        )
        return {
            "r2": r2,
            "weights": {ch: float(w) for ch, w in zip(channels, weights)},
        }

    # ── Combining predictions ────────────────────────────────────────────────

    def combine(
        self,
        channel_predictions: List[ChannelPrediction],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine multi-channel predictions into a single alpha signal.

        Parameters
        ----------
        channel_predictions : list of ChannelPrediction objects

        Returns
        -------
        (direction, conviction) tuple — each of shape (n_stocks,)
        """
        if len(channel_predictions) == 0:
            raise ValueError("No channel predictions to combine.")

        if len(channel_predictions) == 1:
            # Single channel: pass through directly
            cp = channel_predictions[0]
            return cp.direction.copy(), cp.conviction.copy()

        n_stocks = channel_predictions[0].direction.shape[0]

        # Weighted combination of directions
        total_weight = 0.0
        combined_dir = np.zeros(n_stocks, dtype=np.float32)
        combined_conv = np.zeros(n_stocks, dtype=np.float32)

        for cp in channel_predictions:
            w = self._channels.get(cp.name, cp.weight)
            # Weight by both learned weight and channel's own conviction
            effective_weight = abs(w) * cp.conviction
            combined_dir += w * cp.direction
            combined_conv += effective_weight
            total_weight += abs(w)

        if total_weight > 0:
            combined_dir /= total_weight
            combined_conv /= total_weight

        # Clip to valid ranges
        combined_dir = np.clip(combined_dir, -3.0, 3.0)
        combined_conv = np.clip(combined_conv, 0.0, 1.0)

        return combined_dir, combined_conv

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        import pickle
        state = {
            "channels": self._channels,
            "ridge_alpha": self.ridge_alpha,
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._channels = state["channels"]
        self.ridge_alpha = state["ridge_alpha"]
        self._is_trained = state["is_trained"]
        logger.info("Meta-model loaded: channels=%s", list(self._channels.keys()))

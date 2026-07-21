"""
alpha/lightgbm_alpha.py
═══════════════════════
LightGBM-based cross-sectional alpha prediction model.

Predicts forward N-day returns for each stock, producing an AlphaSignal
that the SAC execution agent consumes as part of its observation vector.

Architecture:
  - Input:  50+ alpha features per stock per timestep
  - Output: AlphaSignal = (predicted_return, conviction) per stock
  - Training: Purged walk-forward CV to prevent look-ahead bias
  - Scoring: IC (information coefficient), rank IC, hit rate

The AlphaSignal struct:
  direction  : predicted return sign and magnitude (standardised)
  conviction : model confidence = 1 - normalised prediction uncertainty
               Estimated via variance across CV fold predictions.

Key design decisions:
  - Purged CV with embargo gap eliminates overlap between train labels
    and test features (critical when labels use forward returns).
  - Model is retrained at each walk-forward fold, matching the backtest
    cadence of the SAC agent.
  - Feature importance tracked for interpretability and pruning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    logger.warning(
        "lightgbm not installed. Install with: pip install lightgbm. "
        "LightGBMAlpha will fall back to random predictions."
    )
    HAS_LGB = False

# Fix macOS ARM segfault: LightGBM + PyTorch both link libomp.
# Forcing nthreads=1 for LGB avoids the threading conflict that causes SIGSEGV.
import os as _os
import platform as _platform
if _platform.system() == "Darwin" and _platform.machine() == "arm64":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")


@dataclass
class AlphaSignal:
    """
    Per-stock alpha signal produced by the LightGBM model.

    Attributes
    ----------
    direction   : np.ndarray of shape (n_stocks,)
                  Standardised predicted return. Positive = bullish.
    conviction  : np.ndarray of shape (n_stocks,)
                  Model confidence in [0, 1]. Higher = more certain.
    raw_pred    : np.ndarray of shape (n_stocks,)
                  Raw predicted forward return (un-standardised).
    """
    direction: np.ndarray
    conviction: np.ndarray
    raw_pred: Optional[np.ndarray] = field(default=None)
    feature_importance: Optional[Dict[str, float]] = field(default=None, repr=False)

    def to_obs_vector(self) -> np.ndarray:
        """Flatten to observation vector for the SAC agent: [direction | conviction]."""
        return np.concatenate([self.direction, self.conviction]).astype(np.float32)

    @property
    def n_stocks(self) -> int:
        return len(self.direction)


class LightGBMAlpha:
    """
    LightGBM alpha model with purged walk-forward cross-validation.

    Usage
    -----
    model = LightGBMAlpha(config)
    model.train(X_train, y_train, dates_train)
    signal = model.predict(X_test)  # returns AlphaSignal
    """

    # Default LightGBM params tuned for financial return prediction
    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,        # Column subsampling
        "bagging_fraction": 0.8,         # Row subsampling
        "bagging_freq": 5,
        "min_child_samples": 50,         # Conservative to prevent overfitting
        "lambda_l1": 0.1,               # L1 regularisation
        "lambda_l2": 1.0,               # L2 regularisation
        "max_depth": 7,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "num_threads": 1,               # Prevent libomp segfault on macOS ARM
        "verbose": -1,
    }

    def __init__(self, config: dict):
        self.cfg = config
        alpha_cfg = config.get("alpha", {})

        # Merge user config with defaults
        self.params = {**self.DEFAULT_PARAMS}
        if "lgb_params" in alpha_cfg:
            self.params.update(alpha_cfg["lgb_params"])
        seed = int(config.get("experiment", {}).get("seed", 42))
        self.params.setdefault("seed", seed)
        self.params.setdefault("feature_fraction_seed", seed)
        self.params.setdefault("bagging_seed", seed)
        self.params.setdefault("data_random_seed", seed)

        self.forward_horizon = alpha_cfg.get("forward_horizon", 5)
        self.n_cv_folds = alpha_cfg.get("n_cv_folds", 5)
        self.purge_gap = alpha_cfg.get("purge_gap", 5)

        # Feature pruning config (experiment C)
        self.auto_prune = alpha_cfg.get("auto_prune", False)
        self.max_features = alpha_cfg.get("max_features", 30)
        self.prune_retrain = alpha_cfg.get("prune_retrain", True)

        # Model state
        self._models: List = []           # One model per CV fold
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._pruned_features: Optional[List[str]] = None  # Set after pruning
        self._is_trained = False

        # Prediction uncertainty (std across fold models)
        self._pred_std_scaler = 1.0

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, float]:
        """
        Train LightGBM with purged walk-forward CV.

        Parameters
        ----------
        X         : Feature matrix (n_samples, n_features)
        y         : Forward return labels
        dates     : Date series for purged CV split generation
        cv_splits : Pre-computed CV splits (overrides dates-based generation)

        Returns
        -------
        Dict of evaluation metrics: {ic, rank_ic, hit_rate, mae}
        """
        if not HAS_LGB:
            logger.warning("LightGBM not available — using random fallback.")
            self._is_trained = True
            self._feature_names = list(X.columns)
            return {"ic": 0.0, "rank_ic": 0.0, "hit_rate": 0.5, "mae": 0.0}

        self._feature_names = list(X.columns)
        self._models = []

        # Generate CV splits if not provided
        if cv_splits is None and dates is not None:
            from alpha.feature_engineering import AlphaFeatureEngine
            engine = AlphaFeatureEngine(self.cfg)
            cv_splits = engine.build_purged_cv_splits(
                dates, n_splits=self.n_cv_folds, purge_gap=self.purge_gap
            )

        if cv_splits is None or len(cv_splits) == 0:
            # Fallback: single train/test split (80/20)
            split_idx = int(len(X) * 0.8)
            cv_splits = [(np.arange(split_idx), np.arange(split_idx, len(X)))]

        # Collect OOS predictions for metrics
        all_preds = []
        all_actuals = []
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            callbacks = [lgb.log_evaluation(period=0)]  # Suppress per-iteration logs
            if self.params.get("early_stopping_rounds"):
                callbacks.append(
                    lgb.early_stopping(self.params["early_stopping_rounds"])
                )

            # Remove non-LGB params before training
            train_params = {
                k: v for k, v in self.params.items()
                if k not in ("early_stopping_rounds", "n_estimators")
            }

            model = lgb.train(
                train_params,
                train_data,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=[valid_data],
                callbacks=callbacks,
            )
            self._models.append(model)

            # OOS predictions
            preds = model.predict(X_test)
            all_preds.extend(preds)
            all_actuals.extend(y_test.values)

            # Per-fold metrics
            ic = np.corrcoef(preds, y_test.values)[0, 1] if len(preds) > 1 else 0
            fold_metrics.append({"fold": fold_idx, "ic": ic, "n_test": len(y_test)})
            logger.info(
                "Fold %d: IC=%.4f, n_train=%d, n_test=%d, n_trees=%d",
                fold_idx, ic, len(X_train), len(X_test),
                model.num_trees(),
            )

        # Aggregate metrics
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        metrics = self._compute_metrics(all_preds, all_actuals)
        metrics["n_folds"] = len(cv_splits)

        # Aggregate feature importance across folds
        self._aggregate_feature_importance()

        # Compute prediction std scaler for conviction estimation
        if len(self._models) > 1:
            # Use variance across fold models on a sample
            sample_X = X.iloc[-min(500, len(X)):]
            fold_preds = np.array([m.predict(sample_X) for m in self._models])
            self._pred_std_scaler = float(np.mean(fold_preds.std(axis=0)))
            if self._pred_std_scaler == 0:
                self._pred_std_scaler = 1.0

        self._is_trained = True

        logger.info(
            "Training complete: IC=%.4f, Rank_IC=%.4f, Hit_Rate=%.2f%%, MAE=%.6f",
            metrics["ic"], metrics["rank_ic"],
            metrics["hit_rate"] * 100, metrics["mae"],
        )

        # ── Auto-prune: drop low-importance features, optionally retrain ─────
        if self.auto_prune and HAS_LGB and self._feature_importance:
            pruned_metrics = self._run_auto_prune(X, y, dates, cv_splits, metrics)
            if pruned_metrics is not None:
                metrics = pruned_metrics

        return metrics

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> AlphaSignal:
        """
        Generate AlphaSignal for the given feature matrix.

        Parameters
        ----------
        X : Feature matrix of shape (n_stocks, n_features)
            One row per stock at the current timestep.

        Returns
        -------
        AlphaSignal with direction, conviction, and raw predictions
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        n_stocks = len(X)

        # If auto-prune was used, select only the pruned features
        if self._pruned_features is not None:
            available = [f for f in self._pruned_features if f in X.columns]
            if len(available) == len(self._pruned_features):
                X = X[self._pruned_features]
            else:
                logger.warning(
                    "Auto-prune: expected %d features, found %d. Using full X.",
                    len(self._pruned_features), len(available),
                )

        if not HAS_LGB or len(self._models) == 0:
            # Random fallback
            return AlphaSignal(
                direction=np.zeros(n_stocks, dtype=np.float32),
                conviction=np.full(n_stocks, 0.5, dtype=np.float32),
                raw_pred=np.zeros(n_stocks, dtype=np.float32),
            )

        # Predict with each fold model
        fold_preds = np.array([m.predict(X) for m in self._models])

        # Ensemble: mean prediction
        raw_pred = fold_preds.mean(axis=0)

        # Direction: standardise predictions to zero mean, unit std
        pred_mean = raw_pred.mean()
        pred_std = raw_pred.std()
        if pred_std > 0:
            direction = (raw_pred - pred_mean) / pred_std
        else:
            direction = np.zeros(n_stocks)
        direction = np.clip(direction, -3, 3)  # Cap at ±3σ

        # Conviction: inverse of prediction uncertainty
        # Higher agreement across folds → higher conviction
        if len(self._models) > 1:
            pred_uncertainty = fold_preds.std(axis=0)
            # Normalise by global scaler
            norm_uncertainty = pred_uncertainty / self._pred_std_scaler
            conviction = np.clip(1.0 - norm_uncertainty, 0.0, 1.0)
        else:
            conviction = np.full(n_stocks, 0.5)

        return AlphaSignal(
            direction=direction.astype(np.float32),
            conviction=conviction.astype(np.float32),
            raw_pred=raw_pred.astype(np.float32),
            feature_importance=self._feature_importance,
        )

    def predict_single_step(
        self,
        data: Dict[str, pd.DataFrame],
        date,
        feature_names: List[str],
    ) -> AlphaSignal:
        """
        Convenience method: build feature matrix for a single date and predict.

        Parameters
        ----------
        data           : dict of {ticker: DataFrame} with alpha features
        date           : the date index to predict for
        feature_names  : ordered feature column names

        Returns
        -------
        AlphaSignal
        """
        rows = []
        valid_tickers = []
        configured = self.cfg.get("market", {}).get("tickers", [])
        tickers = [ticker for ticker in configured if ticker in data]
        tickers.extend(ticker for ticker in data if ticker not in tickers)
        for ticker in tickers:
            df = data[ticker]
            if date in df.index:
                rows.append(df.loc[date, feature_names].values)
                valid_tickers.append(ticker)
            else:
                rows.append(np.zeros(len(feature_names)))
                valid_tickers.append(ticker)

        X = pd.DataFrame(rows, columns=feature_names, index=valid_tickers)
        return self.predict(X)

    # ── Feature importance ───────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Return top-N features by importance (gain)."""
        sorted_feats = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_feats[:top_n]

    def _aggregate_feature_importance(self):
        """Average feature importance across all fold models."""
        if not HAS_LGB or not self._models:
            return

        importance_sums: Dict[str, float] = {}
        for model in self._models:
            imp = model.feature_importance(importance_type="gain")
            names = model.feature_name()
            for name, val in zip(names, imp):
                importance_sums[name] = importance_sums.get(name, 0) + val

        n = len(self._models)
        self._feature_importance = {
            k: v / n for k, v in importance_sums.items()
        }

    def _run_auto_prune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series],
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]],
        pre_prune_metrics: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """
        Auto-prune low-importance features and optionally retrain.

        Selects top `max_features` by LightGBM gain importance from the
        initial training run. If `prune_retrain` is True, retrains the
        model on the pruned feature set and returns updated metrics.

        Returns
        -------
        Updated metrics dict if retrained, None otherwise.
        """
        # Rank features by importance
        ranked = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        n_total = len(ranked)
        n_keep = min(self.max_features, n_total)

        if n_keep >= n_total:
            logger.info("Auto-prune: all %d features kept (max_features=%d)", n_total, self.max_features)
            return None

        top_features = [name for name, _ in ranked[:n_keep]]
        dropped = [name for name, _ in ranked[n_keep:]]
        self._pruned_features = top_features

        logger.info(
            "Auto-prune: keeping %d / %d features (dropped: %s)",
            n_keep, n_total,
            ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else ""),
        )

        if not self.prune_retrain:
            logger.info("Auto-prune: prune_retrain=False, skipping retrain")
            return None

        # Retrain on pruned feature set
        logger.info("Auto-prune: retraining on %d pruned features...", n_keep)
        X_pruned = X[top_features]

        # Reset model state for retrain
        self._models = []
        self._feature_names = top_features
        self._feature_importance = {}

        # Re-run the training loop on pruned features
        all_preds = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X_pruned.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_pruned.iloc[test_idx]
            y_test = y.iloc[test_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            callbacks = [lgb.log_evaluation(period=0)]
            if self.params.get("early_stopping_rounds"):
                callbacks.append(
                    lgb.early_stopping(self.params["early_stopping_rounds"])
                )

            train_params = {
                k: v for k, v in self.params.items()
                if k not in ("early_stopping_rounds", "n_estimators")
            }

            model = lgb.train(
                train_params,
                train_data,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=[valid_data],
                callbacks=callbacks,
            )
            self._models.append(model)

            preds = model.predict(X_test)
            all_preds.extend(preds)
            all_actuals.extend(y_test.values)

            ic = np.corrcoef(preds, y_test.values)[0, 1] if len(preds) > 1 else 0
            logger.info(
                "Prune fold %d: IC=%.4f, n_features=%d",
                fold_idx, ic, n_keep,
            )

        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        metrics = self._compute_metrics(all_preds, all_actuals)
        metrics["n_folds"] = len(cv_splits)
        metrics["n_features_pruned"] = n_keep

        # Re-aggregate importance on pruned model
        self._aggregate_feature_importance()

        # Update prediction std scaler
        if len(self._models) > 1:
            sample_X = X_pruned.iloc[-min(500, len(X_pruned)):]
            fold_preds = np.array([m.predict(sample_X) for m in self._models])
            self._pred_std_scaler = float(np.mean(fold_preds.std(axis=0)))
            if self._pred_std_scaler == 0:
                self._pred_std_scaler = 1.0

        logger.info(
            "Auto-prune retrain complete: IC=%.4f → %.4f, features: %d → %d",
            pre_prune_metrics.get("ic", 0), metrics["ic"],
            n_total, n_keep,
        )
        return metrics

    # ── Evaluation metrics ───────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard alpha model evaluation metrics."""
        valid = ~(np.isnan(predictions) | np.isnan(actuals))
        preds = predictions[valid]
        acts = actuals[valid]

        if len(preds) < 2:
            return {"ic": 0, "rank_ic": 0, "hit_rate": 0.5, "mae": 0}

        # Information Coefficient (Pearson correlation)
        ic = float(np.corrcoef(preds, acts)[0, 1])

        # Rank IC (Spearman correlation)
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(preds, acts)
        rank_ic = float(rank_ic) if not np.isnan(rank_ic) else 0.0

        # Hit rate: fraction where sign(pred) == sign(actual)
        hit_rate = float(np.mean(np.sign(preds) == np.sign(acts)))

        # MAE
        mae = float(np.mean(np.abs(preds - acts)))

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "hit_rate": hit_rate,
            "mae": mae,
        }

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save all fold models and metadata."""
        import pickle
        state = {
            "models": self._models,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "pruned_features": self._pruned_features,
            "pred_std_scaler": self._pred_std_scaler,
            "params": self.params,
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Alpha model saved to %s", path)

    def load(self, path: str):
        """Load fold models and metadata."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._models = state["models"]
        self._feature_names = state["feature_names"]
        self._feature_importance = state["feature_importance"]
        self._pruned_features = state.get("pruned_features")
        self._pred_std_scaler = state["pred_std_scaler"]
        self._is_trained = state["is_trained"]
        logger.info("Alpha model loaded from %s (%d folds)", path, len(self._models))

"""
strategy/term_selector.py
──────────────────────────
Determines the trading term (short / medium / long) and manages the
environment and agent configuration for each horizon.

In the first version the term is set by the user at training/inference
time. A future iteration could allow the LLM orchestration layer to
dynamically switch terms based on market regime detection.

Also provides utility functions for building train/eval environment pairs
for a given (risk_profile, term) combination.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.data_loader import NSEDataLoader
from env.trading_env import NSETradingEnv

logger = logging.getLogger(__name__)

# Lazy imports for alpha/temporal/gnn/vae models (only used when enabled)
_AlphaFeatureEngine = None
_LightGBMAlpha = None
_PatchTSTAlpha = None
_TemporalFeatureBuilder = None
_GNNAlpha = None
_RegimeDetector = None


def _import_alpha():
    """Lazy import to avoid circular imports and missing dependency errors."""
    global _AlphaFeatureEngine, _LightGBMAlpha
    if _AlphaFeatureEngine is None:
        from alpha.feature_engineering import AlphaFeatureEngine as AFE
        from alpha.lightgbm_alpha import LightGBMAlpha as LGA
        _AlphaFeatureEngine = AFE
        _LightGBMAlpha = LGA
    return _AlphaFeatureEngine, _LightGBMAlpha


def _import_patchtst():
    """Lazy import for PatchTST modules."""
    global _PatchTSTAlpha, _TemporalFeatureBuilder
    if _PatchTSTAlpha is None:
        from alpha.patchtst import PatchTSTAlpha as PTA
        from alpha.temporal_features import TemporalFeatureBuilder as TFB
        _PatchTSTAlpha = PTA
        _TemporalFeatureBuilder = TFB
    return _PatchTSTAlpha, _TemporalFeatureBuilder


def _import_gnn():
    """Lazy import for GNN module."""
    global _GNNAlpha
    if _GNNAlpha is None:
        from alpha.gnn_alpha import GNNAlpha as GA
        _GNNAlpha = GA
    return _GNNAlpha


def _import_regime():
    """Lazy import for VAE regime module."""
    global _RegimeDetector
    if _RegimeDetector is None:
        from alpha.vae_regime import RegimeDetector as RD, build_market_state, generate_pseudo_labels
        _RegimeDetector = RD
    return _RegimeDetector


VALID_TERMS        = ("short", "medium", "long")
VALID_RISK_PROFILES = ("conservative", "balanced", "aggressive")


class TermSelector:
    """
    Manages environment construction for each term-risk combination.

    Parameters
    ----------
    config     : full config dict
    data_cache : optional pre-loaded data dict to avoid re-downloading
    """

    def __init__(self, config: dict, data_cache: Optional[Dict] = None):
        self.config     = config
        self.loader     = NSEDataLoader(config)
        self.data_cache: Dict = data_cache or {}
        # v3.1 P1: Alpha model state
        self._alpha_enabled = config.get("alpha", {}).get("enabled", False)
        self._alpha_model = None
        self._alpha_engine = None
        self._alpha_features = None  # {ticker: DataFrame} with alpha features
        # v3.1 P2: PatchTST state
        self._patchtst_enabled = config.get("patchtst", {}).get("enabled", False)
        self._patchtst_model = None
        self._temporal_builder = None
        # v3.1 P3: GNN + VAE state
        self._gnn_enabled = config.get("gnn", {}).get("enabled", False)
        self._gnn_model = None
        self._regime_enabled = config.get("vae_regime", {}).get("enabled", False)
        self._regime_model = None
        self._meta_model = None
        self._meta_channel_names = ()
        # Phase 4: Debate analyst
        self._debate_enabled = config.get("llm", {}).get("debate", {}).get("enabled", False)
        self._debate_analyst = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build_env(
        self,
        risk_profile: str,
        term: str,
        train_start: str,
        train_end: str,
        sentiment_fn=None,
        sentiment_fn_eval=None,
        normalise_obs: bool = True,
    ) -> Tuple[DummyVecEnv, DummyVecEnv]:
        """
        Build a (train_env, eval_env) pair for the given risk/term combination.

        The eval environment uses the last 20% of the date range for
        out-of-sample evaluation. Both envs share the same underlying data
        (no data leakage — the split is by index, not random sampling).

        v3.1: If alpha model is enabled, trains the LightGBM alpha model
        on the training split and provides an alpha_fn to both envs.

        Parameters
        ----------
        risk_profile  : "conservative" | "balanced" | "aggressive"
        term          : "short" | "medium" | "long"
        train_start   : ISO date string, e.g. "2018-01-01"
        train_end     : ISO date string, e.g. "2023-01-01"
        sentiment_fn  : optional callable for LLM sentiment
        normalise_obs : wrap envs in VecNormalize for observation scaling

        Returns
        -------
        (train_env, eval_env) — both are DummyVecEnv instances
        """
        self._validate(risk_profile, term)
        import pandas as pd

        warmup_days = int(
            self.config.get("backtest", {}).get(
                "training_warmup_days", 800
            )
        )
        history_start = (
            pd.Timestamp(train_start) - pd.DateOffset(days=warmup_days)
        ).strftime("%Y-%m-%d")
        data = self._get_data(history_start, train_end, term)
        aligned = self.loader.align_dates(data)
        feature_names = self.loader.get_feature_names()

        # Split dates: 80% train, 20% eval
        train_start_ts = pd.Timestamp(train_start)
        all_dates = [
            date
            for date in sorted(next(iter(aligned.values())).index)
            if date >= train_start_ts
        ]
        split_idx = int(len(all_dates) * 0.8)
        train_dates = all_dates[:split_idx]
        eval_dates  = all_dates[split_idx:]

        train_data = {t: df.loc[train_dates] for t, df in aligned.items()}
        eval_data  = {t: df.loc[eval_dates]  for t, df in aligned.items()}

        # v3.1 P1: Build and train alpha model if enabled
        alpha_fn = None
        if self._alpha_enabled:
            alpha_fn = self._build_alpha_pipeline(
                train_data=aligned,
                train_start_date=train_start_ts,
                train_end_date=train_dates[-1] if train_dates else None,
            )

        # v3.1 P2: Build and train PatchTST if enabled
        temporal_fn = None
        if self._patchtst_enabled:
            temporal_fn = self._build_patchtst_pipeline(
                train_data=aligned,
                train_start_date=train_start_ts,
                train_end_date=train_dates[-1] if train_dates else None,
            )

        # v3.1 P3: Build GNN pipeline if enabled
        gnn_fn = None
        if self._gnn_enabled:
            gnn_fn = self._build_gnn_pipeline(
                train_data=aligned,
                train_start_date=train_start_ts,
                train_end_date=train_dates[-1] if train_dates else None,
            )

        # v3.1 P3: Build VAE regime pipeline if enabled
        regime_fn = None
        if self._regime_enabled:
            regime_fn = self._build_regime_pipeline(
                train_data=aligned,
                train_start_date=train_start_ts,
                train_end_date=train_dates[-1] if train_dates else None,
            )

        # v3.2 P0: Combine channels through meta-model if enabled
        meta_model, combined_fn = self._train_meta_model(
            train_data=aligned,
            train_start_date=train_start_ts,
            train_end_date=train_dates[-1] if train_dates else None,
            alpha_fn=alpha_fn,
            temporal_fn=temporal_fn,
            gnn_fn=gnn_fn,
            regime_fn=regime_fn,
        )
        if combined_fn is not None:
            alpha_fn   = combined_fn
            temporal_fn = None
            gnn_fn      = None
            # Regime is a state embedding, not a directional forecast. Keep it
            # as a separate observation instead of coercing it into Ridge.
            logger.info("Meta-model active — directional channels combined into alpha_fn.")

        def make_train():
            return NSETradingEnv(
                data=train_data,
                feature_names=feature_names,
                config=self.config,
                risk_profile=risk_profile,
                term=term,
                sentiment_fn=sentiment_fn,
                alpha_fn=alpha_fn,
                temporal_fn=temporal_fn,
                gnn_fn=gnn_fn,
                regime_fn=regime_fn,
                debate_fn=None,  # disabled during SAC training for speed
            )

        def make_eval():
            return NSETradingEnv(
                data=eval_data,
                feature_names=feature_names,
                config=self.config,
                risk_profile=risk_profile,
                term=term,
                sentiment_fn=sentiment_fn_eval,
                alpha_fn=alpha_fn,
                temporal_fn=temporal_fn,
                gnn_fn=gnn_fn,
                regime_fn=regime_fn,
                debate_fn=None,  # must match train env — obs_dims must be identical
                terminate_on_stop=False,
            )

        train_env = DummyVecEnv([make_train])
        eval_env  = DummyVecEnv([make_eval])

        if normalise_obs:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, clip_obs=10.0,
                                     training=False)
            # Share normalisation statistics from train → eval
            eval_env.obs_rms   = train_env.obs_rms
            eval_env.ret_rms   = train_env.ret_rms

        logger.info(
            "Environments built | risk=%s | term=%s | train=%d | eval=%d | "
            "alpha=%s | patchtst=%s | gnn=%s | regime=%s",
            risk_profile, term, len(train_dates), len(eval_dates),
            "ON" if self._alpha_enabled else "OFF",
            "ON" if self._patchtst_enabled else "OFF",
            "ON" if self._gnn_enabled else "OFF",
            "ON" if self._regime_enabled else "OFF",
        )

        # Cache trained fns so build_backtest_env can reuse them for the same fold
        self._trained_fns = {
            "alpha_fn":    alpha_fn,
            "temporal_fn": temporal_fn,
            "gnn_fn":      gnn_fn,
            "regime_fn":   regime_fn,
            "train_end":   train_end,
            "fold_id":     (train_start, train_end),
        }

        return train_env, eval_env

    def build_backtest_env(
        self,
        risk_profile: str,
        term: str,
        start: str,
        end: str,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        sentiment_fn=None,
        warmup_days: int = 400,
        enable_debate: bool = False,
    ) -> NSETradingEnv:
        """
        Build a single un-vectorised environment for backtesting.

        Downloads `warmup_days` of extra data before `start` so that
        technical indicators (50-day EMA, MACD, Bollinger Bands, etc.)
        are fully warm when the test period begins. The environment only
        steps through dates from `start` onwards — the warmup data is
        used purely for indicator computation.
        """
        self._validate(risk_profile, term)
        import pandas as pd

        # Extend download window backwards to warm up indicators
        warmup_start = (
            pd.Timestamp(start) - pd.DateOffset(days=warmup_days)
        ).strftime("%Y-%m-%d")

        # When invoked from walk-forward evaluation, load the complete fold
        # history through the OOS end date. Models remain fitted only through
        # train_end, while inference functions can access actual test-date
        # features and causal lookbacks.
        data_start = train_start or warmup_start
        data = self._get_data(data_start, end, term)
        aligned = self.loader.align_dates(data)

        # Slice each ticker to only trade from the original start date,
        # but keep pre-start rows available for the env's price lookups.
        # We trim to start date so the env steps only through the test window.
        test_start_ts = pd.Timestamp(start)
        trimmed = {
            ticker: df[df.index >= test_start_ts]
            for ticker, df in aligned.items()
        }

        # Warn if warmup didn't help enough
        min_bars = min(len(df) for df in trimmed.values())
        if min_bars < 20:
            logger.warning(
                "Only %d bars in test window after warmup trim — "
                "consider a longer test period or more warmup days.", min_bars
            )

        feature_names = self.loader.get_feature_names()

        logger.info(
            "Backtest env | risk=%s | term=%s | test_start=%s | bars=%d",
            risk_profile, term, start, min_bars,
        )

        cached = getattr(self, "_trained_fns", None)
        expected_fold_id = (
            (train_start, train_end) if train_start is not None and train_end is not None else None
        )
        use_cached = False
        if cached is not None:
            cached_fold_id = cached.get("fold_id")
            if expected_fold_id is None:
                use_cached = True
            elif cached_fold_id == expected_fold_id:
                use_cached = True
            else:
                # Safety over stale reuse: stale channel fns quietly corrupt backtests.
                logger.warning(
                    "Backtest cache fold mismatch: cached=%s, requested=%s. "
                    "Ignoring cache and rebuilding channels from standalone path.",
                    cached_fold_id,
                    expected_fold_id,
                )

        if use_cached and cached is not None:
            # Rebind fitted models to the full causal panel. Reusing closures
            # from build_env would leave every later OOS date absent.
            alpha_fn, temporal_fn, gnn_fn, regime_fn = (
                self._bind_trained_channel_fns(aligned)
            )
            logger.info(
                "Backtest: rebound fold-trained channels to OOS feature data."
            )
        else:
            # Standalone backtest path — rebuild from warmup data (requires warmup_days >= 400)
            alpha_fn = None
            if self._alpha_enabled:
                alpha_fn = self._build_alpha_pipeline(
                    train_data=aligned,
                    train_start_date=train_start,
                    train_end_date=test_start_ts,
                )

            temporal_fn = None
            if self._patchtst_enabled:
                temporal_fn = self._build_patchtst_pipeline(
                    train_data=aligned,
                    train_start_date=train_start,
                    train_end_date=test_start_ts,
                )

            gnn_fn = None
            if self._gnn_enabled:
                gnn_fn = self._build_gnn_pipeline(
                    train_data=aligned,
                    train_start_date=train_start,
                    train_end_date=test_start_ts,
                )

            regime_fn = None
            if self._regime_enabled:
                regime_fn = self._build_regime_pipeline(
                    train_data=aligned,
                    train_start_date=train_start,
                    train_end_date=test_start_ts,
                )

            _, combined_fn = self._train_meta_model(
                train_data=aligned,
                train_start_date=train_start,
                train_end_date=test_start_ts,
                alpha_fn=alpha_fn,
                temporal_fn=temporal_fn,
                gnn_fn=gnn_fn,
                regime_fn=regime_fn,
            )
            if combined_fn is not None:
                alpha_fn    = combined_fn
                temporal_fn = None
                gnn_fn      = None
                logger.info("Backtest (standalone): meta-model active.")

        # Phase 4: debate only in analysis runs.
        # Keep disabled by default so backtest obs_dim matches train obs_dim.
        debate_fn = None
        if enable_debate and self._debate_enabled:
            debate_fn = self._build_debate_pipeline()

        return NSETradingEnv(
            data=trimmed,
            feature_names=feature_names,
            config=self.config,
            risk_profile=risk_profile,
            term=term,
            sentiment_fn=sentiment_fn,
            alpha_fn=alpha_fn,
            temporal_fn=temporal_fn,
            gnn_fn=gnn_fn,
            regime_fn=regime_fn,
            debate_fn=debate_fn,
            terminate_on_stop=False,
        )

    def term_description(self, term: str) -> dict:
        """Return human-readable description of a term's parameters."""
        term_cfg = self.config["terms"][term]
        descriptions = {
            "short":  "Intraday to 1-day holds. High frequency, focuses on momentum and mean reversion.",
            "medium": "1–4 week holds. Balances trend following with fundamental catalysts.",
            "long":   "1–3 month holds. Macro-driven, lower turnover, valuation-oriented.",
        }
        return {
            "term":        term,
            "description": descriptions[term],
            "interval":    term_cfg["interval"],
            "lookback":    f"{term_cfg['lookback_days']} days",
            "hold_period": f"up to {term_cfg['hold_period_days']} days",
            "rebalance":   term_cfg["rebalance_freq"],
        }

    # ── v3.1 Alpha model integration ────────────────────────────────────────

    def _build_alpha_pipeline(
        self,
        train_data: Dict,
        train_start_date=None,
        train_end_date=None,
    ):
        """
        Build and train the LightGBM alpha model, returning an alpha_fn
        callable that the trading env can invoke at each step.

        Parameters
        ----------
        train_data      : dict of {ticker: DataFrame} with OHLCV
        train_end_date  : training cutoff date (no data after this used for training)

        Returns
        -------
        alpha_fn : callable(date) → AlphaSignal
        """
        AlphaFeatureEngine, LightGBMAlpha = _import_alpha()
        import pandas as pd

        engine = AlphaFeatureEngine(self.config)
        alpha_cfg = self.config.get("alpha", {})

        # Step 1: Compute alpha features on raw OHLCV data
        featured_data = engine.build_features(train_data)

        if not featured_data:
            logger.warning("Alpha feature computation returned no data — disabling alpha.")
            return None

        # Step 2: Z-score normalise features
        featured_data = engine.zscore_features(featured_data)

        # Store for later prediction use
        self._alpha_features = featured_data

        # Step 3: Build training set (only using data up to train_end_date)
        train_start = (
            pd.Timestamp(train_start_date)
            if train_start_date is not None
            else None
        )
        train_cutoff = (
            pd.Timestamp(train_end_date)
            if train_end_date is not None
            else None
        )
        train_subset = {
            ticker: frame[
                (frame.index >= train_start if train_start is not None else True)
                & (frame.index <= train_cutoff if train_cutoff is not None else True)
            ]
            for ticker, frame in featured_data.items()
        }

        forward_horizon = alpha_cfg.get("forward_horizon", 5)
        # The embargo must cover the full label horizon; a shorter gap leaves
        # training labels overlapping validation features.
        purge_gap = max(alpha_cfg.get("purge_gap", 5), forward_horizon)

        try:
            X, y = engine.build_training_set(
                train_subset,
                forward_horizon=forward_horizon,
                purge_gap=purge_gap,
            )
        except ValueError as e:
            logger.warning("Alpha training set construction failed: %s", e)
            return None

        # Step 4: Train LightGBM
        model = LightGBMAlpha(self.config)

        # build_training_set preserves the exact date/ticker MultiIndex after
        # dropping invalid rows, so purged CV cannot silently fall back to a
        # ticker-wise row split.
        dates_series = pd.Series(
            X.index.get_level_values("date").to_numpy()
        ).reset_index(drop=True)

        metrics = model.train(X, y, dates=dates_series)
        self._alpha_model = model
        self._alpha_engine = engine

        logger.info(
            "Alpha model trained: IC=%.4f, Rank_IC=%.4f, Hit_Rate=%.1f%%",
            metrics.get("ic", 0), metrics.get("rank_ic", 0),
            metrics.get("hit_rate", 0.5) * 100,
        )

        # Log top features
        top_feats = model.get_feature_importance(top_n=10)
        if top_feats:
            logger.info("Top alpha features: %s",
                        ", ".join(f"{n}({v:.0f})" for n, v in top_feats))

        return self._make_alpha_fn(model, featured_data)

    def _make_alpha_fn(self, model, featured_data: Dict):
        """Bind a fitted LightGBM model to a causal inference feature panel."""
        feature_names = self._alpha_engine.get_feature_names()
        # Experiment B: conviction gate — zero out signals below threshold
        alpha_cfg = self.config.get("alpha", {})
        conviction_gate = alpha_cfg.get("conviction_gate", 0.0)

        # Phase 4: Adversarial conviction filter (disabled during SAC training)
        adv_filter = None
        adv_cfg = alpha_cfg.get("adversarial_filter", {})
        if adv_cfg.get("enabled", False):
            from alpha.adversarial_filter import AdversarialFilter
            adv_filter = AdversarialFilter(
                self.config, self.config["market"]["tickers"]
            )
        logger.info(
            "Alpha post-processing: conviction_gate=%s | adversarial_filter=%s",
            f"{conviction_gate:.3f}" if conviction_gate > 0 else "OFF",
            "ON" if adv_filter is not None else "OFF",
        )

        def alpha_fn(date):
            missing = [
                ticker for ticker, df in featured_data.items()
                if date not in df.index
            ]
            if missing:
                raise KeyError(
                    f"Alpha inference date {date} missing for {missing}"
                )
            signal = model.predict_single_step(
                featured_data, date, feature_names
            )
            if conviction_gate > 0:
                mask = signal.conviction < conviction_gate
                signal.direction[mask] = 0.0
                signal.conviction[mask] = 0.0
            if adv_filter is not None:
                date_str = str(date)[:10]
                signal = adv_filter.filter(signal, date_str)
            return signal

        return alpha_fn

    def _build_patchtst_pipeline(
        self,
        train_data: Dict,
        train_start_date=None,
        train_end_date=None,
    ):
        """
        Build and train the PatchTST temporal model, returning a temporal_fn
        callable for the trading env.

        Parameters
        ----------
        train_data      : dict of {ticker: DataFrame} with OHLCV + indicators
        train_end_date  : training cutoff date

        Returns
        -------
        temporal_fn : callable(date) → TemporalSignal
        """
        PatchTSTAlpha, TemporalFeatureBuilder = _import_patchtst()

        feature_names = self.loader.get_feature_names()
        builder = TemporalFeatureBuilder(self.config, feature_names)
        patchtst_cfg = self.config.get("patchtst", {})
        forward_horizon = patchtst_cfg.get("forward_horizon", 5)

        # Step 1: Filter data to training period only
        import pandas as pd
        train_start = (
            pd.Timestamp(train_start_date)
            if train_start_date is not None
            else None
        )
        train_cutoff = (
            pd.Timestamp(train_end_date)
            if train_end_date is not None
            else None
        )
        train_subset = {
            ticker: frame[
                (frame.index >= train_start if train_start is not None else True)
                & (frame.index <= train_cutoff if train_cutoff is not None else True)
            ]
            for ticker, frame in train_data.items()
        }

        # Step 2: Build training sequences
        try:
            sequences, labels, meta = builder.build_training_set(
                train_subset, forward_horizon=forward_horizon
            )
        except ValueError as e:
            logger.warning("PatchTST sequence construction failed: %s", e)
            return None

        # Step 3: Split and train
        X_train, y_train, X_val, y_val = builder.build_validation_split(
            sequences, labels, meta
        )

        model = PatchTSTAlpha(self.config)
        metrics = model.train(X_train, y_train, X_val, y_val)
        self._patchtst_model = model
        self._temporal_builder = builder

        logger.info(
            "PatchTST trained: val_loss=%.4f, IC=%.4f, best_epoch=%d, params=%s",
            metrics.get("val_loss", 0), metrics.get("ic", 0),
            metrics.get("best_epoch", 0),
            f"{metrics.get('n_params', 0) / 1000:.1f}K",
        )

        return self._make_temporal_fn(model, builder, train_data)

    @staticmethod
    def _make_temporal_fn(model, builder, inference_data: Dict):
        """Bind a fitted PatchTST model to an inference panel."""
        def temporal_fn(date):
            missing = [
                ticker for ticker, df in inference_data.items()
                if date not in df.index
            ]
            if missing:
                raise KeyError(
                    f"Temporal inference date {date} missing for {missing}"
                )
            seq_dict = builder.get_sequences_for_date(inference_data, date)
            return model.predict_single_step(seq_dict)

        return temporal_fn

    # ── v3.1 P3: GNN + VAE pipelines ────────────────────────────────────────

    def _build_gnn_pipeline(
        self,
        train_data: Dict,
        train_start_date=None,
        train_end_date=None,
    ):
        """
        Build and train the GNN, returning a gnn_fn(date) → GraphSignal closure.

        Node features (6 dims per stock):
          - ret_1d, ret_5d, ret_20d (3 dims from alpha features)
          - vol_5d, vol_20d (2 dims)
          - volume_z_20 (1 dim)
        """
        GNNAlpha = _import_gnn()
        import pandas as pd
        import numpy as np

        gnn_cfg = self.config.get("gnn", {})
        forward_horizon = gnn_cfg.get("forward_horizon", 5)
        tickers = self._ordered_tickers(train_data)
        n_stocks = len(tickers)

        # Use alpha features if available, fall back to raw market data
        source_data = self._alpha_features if self._alpha_features else train_data

        # Filter to training period
        start_cutoff = (
            pd.Timestamp(train_start_date)
            if train_start_date is not None
            else None
        )
        end_cutoff = (
            pd.Timestamp(train_end_date)
            if train_end_date is not None
            else None
        )
        source_subset = {
            ticker: frame[
                (frame.index >= start_cutoff if start_cutoff is not None else True)
                & (frame.index <= end_cutoff if end_cutoff is not None else True)
            ]
            for ticker, frame in source_data.items()
        }

        # Collect common dates
        common_dates = None
        for df in source_subset.values():
            idx = set(df.index)
            common_dates = idx if common_dates is None else common_dates & idx
        if not common_dates:
            logger.warning("GNN: no common dates — skipping.")
            return None
        common_dates = sorted(common_dates)

        # Causal market node features. The previous four leading slots were
        # documented as alpha/temporal inputs but were always zero.
        DATA_COLS = ["ret_1d", "ret_5d", "ret_20d", "vol_5d", "vol_20d", "volume_z_20"]
        D_INPUT = len(DATA_COLS)

        # Build training arrays
        n_samples = len(common_dates) - forward_horizon
        if n_samples < 50:
            logger.warning("GNN: insufficient training samples (%d) — skipping.", n_samples)
            return None

        X_list, returns_list, y_list = [], [], []

        for t_idx in range(n_samples):
            date = common_dates[t_idx]
            future_date = common_dates[t_idx + forward_horizon]

            node_feats = []
            rets_today = []
            labels_row = []

            for ticker in tickers:
                df = source_subset.get(ticker, pd.DataFrame())
                row_feats = [0.0] * D_INPUT

                if date in df.index:
                    row = df.loc[date]
                    for fi, col in enumerate(DATA_COLS):
                        v = row.get(col, 0.0) if hasattr(row, "get") else (
                            row[col] if col in df.columns else 0.0
                        )
                        if v is None or (isinstance(v, float) and (v != v)):  # nan check
                            v = 0.0
                        row_feats[fi] = float(v)

                node_feats.append(row_feats)
                rets_today.append(row_feats[0])

                # Label: cumulative return from the current close to the
                # configured future horizon (not the isolated 1-day return
                # observed at that future date).
                ticker_df = source_subset.get(ticker, pd.DataFrame())
                if (
                    date in ticker_df.index
                    and future_date in ticker_df.index
                    and "Close" in ticker_df.columns
                ):
                    current_close = float(ticker_df.loc[date, "Close"])
                    future_close = float(ticker_df.loc[future_date, "Close"])
                    fut_ret = (
                        np.log(future_close / current_close)
                        if current_close > 0 and future_close > 0
                        else 0.0
                    )
                else:
                    fut_ret = 0.0
                labels_row.append(float(fut_ret))

            X_list.append(node_feats)
            returns_list.append(rets_today)
            y_list.append(labels_row)

        X = np.array(X_list, dtype=np.float32)          # (n_samples, n_stocks, D_INPUT)
        returns_arr = np.array(returns_list, dtype=np.float32)  # (n_samples, n_stocks)
        y = np.array(y_list, dtype=np.float32)          # (n_samples, n_stocks)

        # Train GNN
        model = GNNAlpha(self.config)
        metrics = model.train(X, returns_arr, y)
        self._gnn_model = model

        logger.info(
            "GNN trained: train_loss=%.4f, val_loss=%.4f, best_epoch=%d",
            metrics.get("train_loss", 0), metrics.get("val_loss", 0),
            metrics.get("best_epoch", 0),
        )

        return self._make_gnn_fn(model, source_data)

    def _make_gnn_fn(self, model, source_data: Dict):
        """Bind a fitted GNN to current-date node features and graph history."""
        import numpy as np
        import pandas as pd

        tickers = self._ordered_tickers(source_data)
        data_cols = [
            "ret_1d", "ret_5d", "ret_20d",
            "vol_5d", "vol_20d", "volume_z_20",
        ]
        d_input = len(data_cols)
        window = self.config.get("gnn", {}).get("correlation_window", 60)
        common_dates = sorted(
            set.intersection(*(set(source_data[t].index) for t in tickers))
        )
        date_positions = {date: idx for idx, date in enumerate(common_dates)}

        def gnn_fn(date):
            if date not in date_positions:
                raise KeyError(f"GNN inference date {date} missing from feature panel")
            node_feats = []
            ret_rows = []

            for ticker in tickers:
                df = source_data.get(ticker, pd.DataFrame())
                row_feats = [0.0] * d_input

                if date in df.index:
                    row = df.loc[date]
                    for fi, col in enumerate(data_cols):
                        v = row.get(col, 0.0) if hasattr(row, "get") else (
                            row[col] if col in df.columns else 0.0
                        )
                        if v is None or not np.isfinite(float(v)):
                            v = 0.0
                        row_feats[fi] = float(v)

                node_feats.append(row_feats)
                ret_rows.append(row_feats[0])

            node_features_np = np.array(node_feats, dtype=np.float32)

            # Build rolling returns window for adjacency
            idx_pos = date_positions[date]
            start = max(0, idx_pos - window + 1)
            window_dates = common_dates[start:idx_pos + 1]
            rets_window = []
            for window_date in window_dates:
                row_rets = []
                for ticker in tickers:
                    df = source_data.get(ticker, pd.DataFrame())
                    value = (
                        float(df.loc[window_date, "ret_1d"])
                        if window_date in df.index and "ret_1d" in df.columns
                        else 0.0
                    )
                    row_rets.append(value if np.isfinite(value) else 0.0)
                rets_window.append(row_rets)
            returns_window = np.array(rets_window, dtype=np.float32)

            return model.predict(node_features_np, returns_window)

        return gnn_fn

    def _build_regime_pipeline(
        self,
        train_data: Dict,
        train_start_date=None,
        train_end_date=None,
    ):
        """
        Build and train the VAE regime detector, returning a regime_fn(date) → RegimeSignal closure.
        """
        RegimeDetectorCls = _import_regime()
        from alpha.vae_regime import build_market_state, generate_pseudo_labels
        import pandas as pd
        import numpy as np

        tickers = self._ordered_tickers(train_data)

        # Filter to training period
        start_cutoff = (
            pd.Timestamp(train_start_date)
            if train_start_date is not None
            else None
        )
        end_cutoff = (
            pd.Timestamp(train_end_date)
            if train_end_date is not None
            else None
        )
        train_subset = {
            ticker: frame[
                (frame.index >= start_cutoff if start_cutoff is not None else True)
                & (frame.index <= end_cutoff if end_cutoff is not None else True)
            ]
            for ticker, frame in train_data.items()
        }

        # Common dates
        common_dates = None
        for df in train_subset.values():
            idx = set(df.index)
            common_dates = idx if common_dates is None else common_dates & idx
        if not common_dates:
            logger.warning("VAE: no common dates — skipping.")
            return None
        common_dates = sorted(common_dates)

        # Build market states and pseudo-labels
        states = []
        valid_dates = []
        for date in common_dates:
            try:
                state = build_market_state(train_subset, date, tickers)
                if not np.any(np.isnan(state)):
                    states.append(state)
                    valid_dates.append(date)
            except Exception:
                continue

        if len(states) < 50:
            logger.warning("VAE: insufficient training samples (%d) — skipping.", len(states))
            return None

        X = np.stack(states).astype(np.float32)

        # Generate pseudo-labels
        all_labels = generate_pseudo_labels(train_subset, tickers)
        labels_arr = np.array([all_labels.get(d, 0) for d in valid_dates], dtype=np.int64)

        # Train VAE
        detector = RegimeDetectorCls(self.config)
        metrics = detector.train(X, labels_arr)
        self._regime_model = detector

        logger.info(
            "VAE trained: recon_loss=%.4f, kl_loss=%.4f, best_epoch=%d",
            metrics.get("recon_loss", 0), metrics.get("kl_loss", 0),
            metrics.get("best_epoch", 0),
        )

        return self._make_regime_fn(detector, train_data)

    def _make_regime_fn(self, detector, inference_data: Dict):
        """Bind a fitted regime detector to an inference market panel."""
        from alpha.vae_regime import build_market_state
        import numpy as np

        tickers = self._ordered_tickers(inference_data)
        def regime_fn(date):
            missing = [
                ticker for ticker in tickers
                if date not in inference_data[ticker].index
            ]
            if missing:
                raise KeyError(
                    f"Regime inference date {date} missing for {missing}"
                )
            state = build_market_state(inference_data, date, tickers)
            if not np.all(np.isfinite(state)):
                raise ValueError(f"Non-finite regime state at {date}")
            return detector.predict(state)

        return regime_fn

    def _bind_trained_channel_fns(self, inference_data: Dict):
        """
        Bind fold-fitted channel models to a panel that includes OOS dates.

        Fitting state remains frozen; only deterministic feature/signal lookup
        data changes. This prevents train-window closures from silently emitting
        zero/default signals throughout the following test window.
        """
        alpha_fn = temporal_fn = gnn_fn = regime_fn = None
        alpha_features = None

        if self._alpha_enabled and self._alpha_model is not None:
            engine = self._alpha_engine
            alpha_features = engine.zscore_features(
                engine.build_features(inference_data)
            )
            if not alpha_features:
                raise RuntimeError("No alpha features available for OOS binding.")
            alpha_fn = self._make_alpha_fn(self._alpha_model, alpha_features)

        if self._patchtst_enabled and self._patchtst_model is not None:
            temporal_fn = self._make_temporal_fn(
                self._patchtst_model,
                self._temporal_builder,
                inference_data,
            )

        if self._gnn_enabled and self._gnn_model is not None:
            gnn_fn = self._make_gnn_fn(
                self._gnn_model,
                alpha_features or inference_data,
            )

        if self._regime_enabled and self._regime_model is not None:
            regime_fn = self._make_regime_fn(
                self._regime_model,
                inference_data,
            )

        if getattr(self, "_meta_model", None) is not None:
            channel_fns = {}
            if alpha_fn is not None:
                channel_fns["lgbm_alpha"] = alpha_fn
            if temporal_fn is not None:
                channel_fns["patchtst"] = temporal_fn
            if gnn_fn is not None:
                channel_fns["gnn"] = gnn_fn
            expected = set(getattr(self, "_meta_channel_names", ()))
            if set(channel_fns) != expected:
                raise RuntimeError(
                    "Meta-model channel mismatch during OOS binding: "
                    f"expected={sorted(expected)}, got={sorted(channel_fns)}"
                )
            alpha_fn = self._make_combined_fn(
                self._meta_model,
                channel_fns,
            )
            temporal_fn = None
            gnn_fn = None

        return alpha_fn, temporal_fn, gnn_fn, regime_fn

    def _train_meta_model(
        self,
        train_data: Dict,
        train_start_date=None,
        train_end_date=None,
        alpha_fn=None,
        temporal_fn=None,
        gnn_fn=None,
        regime_fn=None,
    ):
        """
        Train the Ridge meta-model to combine all alpha channels.

        Returns (meta_model, combined_fn) or (None, None) if disabled/insufficient data.
        """
        from alpha.meta_model import RidgeMetaModel, ChannelPrediction
        from alpha.lightgbm_alpha import AlphaSignal
        import pandas as pd
        import numpy as np

        meta_cfg = self.config.get("meta_model", {})
        if not meta_cfg.get("enabled", False):
            return None, None

        ridge_alpha = meta_cfg.get("ridge_alpha", 1.0)
        min_samples = meta_cfg.get("min_train_samples", 200)
        forward_horizon = self.config.get("alpha", {}).get("forward_horizon", 10)

        meta = RidgeMetaModel(
            config=self.config,
            n_stocks=len(self.config["market"]["tickers"]),
            ridge_alpha=ridge_alpha,
        )

        channel_fns = {}
        if alpha_fn is not None:
            meta.register_channel("lgbm_alpha")
            channel_fns["lgbm_alpha"] = alpha_fn
        if temporal_fn is not None:
            meta.register_channel("patchtst")
            channel_fns["patchtst"] = temporal_fn
        if gnn_fn is not None:
            meta.register_channel("gnn")
            channel_fns["gnn"] = gnn_fn
        if len(channel_fns) < 2:
            logger.info("Meta-model skipped — fewer than 2 channels available.")
            return None, None

        ordered_tickers = self._ordered_tickers(train_data)
        ref_ticker = ordered_tickers[0]
        all_dates = sorted(train_data[ref_ticker].index)

        if train_end_date is not None:
            cutoff = pd.Timestamp(train_end_date)
            train_dates = [d for d in all_dates if d <= cutoff]
        else:
            train_dates = all_dates
        if train_start_date is not None:
            start_cutoff = pd.Timestamp(train_start_date)
            train_dates = [d for d in train_dates if d >= start_cutoff]

        if len(train_dates) < min_samples + forward_horizon:
            logger.warning(
                "Not enough training dates for meta-model (%d < %d).",
                len(train_dates), min_samples + forward_horizon,
            )
            return None, None

        sample_dates = train_dates[60:-forward_horizon]

        channel_preds = {name: [] for name in channel_fns}
        actuals = []

        for date in sample_dates:
            try:
                future_idx = all_dates.index(date) + forward_horizon
            except ValueError:
                continue
            if future_idx >= len(all_dates):
                continue
            future_date = all_dates[future_idx]

            fwd_rets = []
            for ticker in ordered_tickers:
                df = train_data[ticker]
                if date in df.index and future_date in df.index:
                    ret = (df.loc[future_date, "Close"] - df.loc[date, "Close"]) / max(df.loc[date, "Close"], 1e-8)
                    fwd_rets.append(float(ret))
                else:
                    fwd_rets.append(0.0)
            actuals.extend(fwd_rets)

            for name, fn in channel_fns.items():
                try:
                    signal = fn(date)
                    direction = np.asarray(
                        getattr(
                            signal,
                            "direction",
                            np.zeros(len(ordered_tickers)),
                        ),
                        dtype=np.float64,
                    )
                    if direction.shape != (len(ordered_tickers),):
                        raise ValueError(
                            f"{name} direction shape {direction.shape} "
                            f"!= ({len(ordered_tickers)},)"
                        )
                    channel_preds[name].extend(direction.tolist())
                except Exception:
                    channel_preds[name].extend(
                        [0.0] * len(ordered_tickers)
                    )

        actuals_arr = np.array(actuals)
        preds_dict = {name: np.array(vals) for name, vals in channel_preds.items()}

        try:
            result = meta.train(preds_dict, actuals_arr)
            logger.info(
                "Meta-model trained: R²=%.4f, weights=%s",
                result["r2"], result["weights"],
            )
        except Exception as e:
            logger.warning("Meta-model training failed: %s — using equal weights.", e)

        self._meta_model = meta
        self._meta_channel_names = tuple(channel_fns)
        return meta, self._make_combined_fn(meta, channel_fns)

    def _make_combined_fn(self, meta, channel_fns: Dict):
        """Bind a fitted Ridge combiner to directional channel callables."""
        from alpha.meta_model import ChannelPrediction
        from alpha.lightgbm_alpha import AlphaSignal
        import numpy as np

        n_stocks = len(self.config["market"]["tickers"])

        def combined_alpha_fn(date):
            predictions = []
            for name, fn in channel_fns.items():
                try:
                    signal = fn(date)
                    direction = np.array(
                        getattr(signal, "direction", np.zeros(n_stocks)),
                        dtype=np.float32,
                    )
                    conviction = np.array(
                        getattr(signal, "conviction", np.zeros(n_stocks)),
                        dtype=np.float32,
                    )
                    predictions.append(ChannelPrediction(name=name, direction=direction, conviction=conviction))
                except Exception:
                    pass

            if predictions:
                combined_dir, combined_conv = meta.combine(predictions)
            else:
                raise RuntimeError(
                    f"No meta-model channel available for date {date}"
                )

            return AlphaSignal(
                direction=combined_dir,
                conviction=combined_conv,
                raw_pred=combined_dir.copy(),
            )

        return combined_alpha_fn

    def _build_debate_pipeline(self):
        """
        Instantiate the DebateAnalyst and return a debate_fn(date) → DebateSignal closure.

        The closure calls the debate analyst with zero alpha signals (no live alpha
        context available at this stage). In backtesting, alpha signals can be injected
        by extending the closure to read from self._alpha_model.
        """
        from llm.debate_analyst import DebateAnalyst
        import numpy as np

        analyst = DebateAnalyst(self.config, list(self.config["market"]["tickers"]))
        self._debate_analyst = analyst
        _analyst = analyst
        _n_stocks = len(self.config["market"]["tickers"])

        def debate_fn(date):
            try:
                date_str = str(date)[:10]  # normalise to YYYY-MM-DD
                alpha_dirs = np.zeros(_n_stocks, dtype=np.float32)
                alpha_convs = np.zeros(_n_stocks, dtype=np.float32)
                return _analyst.analyze(alpha_dirs, alpha_convs, {}, date_str)
            except Exception:
                return _analyst.get_zero_signal()

        return debate_fn

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ordered_tickers(self, data: Dict) -> list[str]:
        """Return tickers in the portfolio/config order used by the environment."""
        configured = self.config.get("market", {}).get("tickers", [])
        tickers = [ticker for ticker in configured if ticker in data]
        tickers.extend(ticker for ticker in data if ticker not in tickers)
        return tickers

    def _get_data(self, start: str, end: str, term: str) -> Dict:
        cache_key = f"{start}_{end}_{term}"
        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = self.loader.load(
                start=start, end=end, term=term
            )
        data = self.data_cache[cache_key]
        self._last_data_fingerprint = self.loader.fingerprint(data)
        return data

    def _validate(self, risk_profile: str, term: str) -> None:
        if risk_profile not in VALID_RISK_PROFILES:
            raise ValueError(f"Invalid risk_profile '{risk_profile}'. Choose from {VALID_RISK_PROFILES}")
        if term not in VALID_TERMS:
            raise ValueError(f"Invalid term '{term}'. Choose from {VALID_TERMS}")

"""
alpha/feature_engineering.py
════════════════════════════
Comprehensive feature engineering for the LightGBM alpha model.

Produces 50+ features per stock per timestep, grouped into:
  1. Momentum & Returns        (12 features)
  2. Volatility & Risk         (8 features)
  3. Volume Anomalies          (6 features)
  4. Trend Indicators          (8 features)
  5. Mean Reversion            (5 features)
  6. Cross-Stock / Relative    (6 features)
  7. Rolling Statistical       (6 features)
  8. Calendar / Structural     (3 features)

All features are computed using ONLY past data — no look-ahead bias.
Forward return labels for supervised training use explicit embargo/purge gaps.

Design decisions:
  - Features are computed on RAW (un-normalised) prices, then z-scored.
  - Cross-stock features use the full panel (all tickers at each date).
  - Forward returns for labels are computed separately with purge gaps.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


# ── Feature group definitions ────────────────────────────────────────────────

MOMENTUM_FEATURES = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "mom_12_1",          # 12-month momentum skipping most recent month
    "roc_5", "roc_20",   # Rate of change
    "williams_r",        # Williams %R
    "stoch_k", "stoch_d",  # Stochastic oscillator
]

VOLATILITY_FEATURES = [
    "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    "vol_ratio_5_20",     # Short-term vs long-term vol
    "atr_14", "atr_ratio",  # ATR and ATR/close
    "parkinson_vol",      # Parkinson high-low volatility estimator
]

VOLUME_FEATURES = [
    "volume_z_20", "volume_z_60",     # Volume z-scores at different horizons
    "obv_slope",                       # On-balance volume trend (5-day slope)
    "vwap_dist",                       # Distance from VWAP proxy
    "volume_breakout",                 # Binary: volume > 2× 20d avg
    "volume_price_corr",               # Rolling corr(volume, abs(return))
]

TREND_FEATURES = [
    "ema_dist_9", "ema_dist_21", "ema_dist_50", "ema_dist_200",
    "adx_14",            # Average Directional Index
    "cci_20",            # Commodity Channel Index
    "macd_hist",         # MACD histogram
    "trend_strength",    # ADX * sign(close - ema_50)
]

MEAN_REVERSION_FEATURES = [
    "rsi_14", "rsi_5",
    "bb_pct", "bb_width",
    "dist_from_52w_high",  # (close - 52w high) / 52w high
]

CROSS_STOCK_FEATURES = [
    "ret_rank",           # Cross-sectional return rank (percentile)
    "vol_rank",           # Cross-sectional volatility rank
    "mom_rank",           # Cross-sectional momentum rank (20d)
    "ret_vs_mean",        # Return minus cross-sectional mean
    "sector_rel_mom",     # Return vs equal-weight basket (proxy for sector)
    "dispersion_contrib", # Contribution to cross-sectional dispersion
]

ROLLING_STAT_FEATURES = [
    "skewness_20", "kurtosis_20",
    "skewness_60", "kurtosis_60",
    "autocorr_1",         # Lag-1 return autocorrelation (20d rolling)
    "max_drawdown_20",    # Rolling 20d max drawdown
]

CALENDAR_FEATURES = [
    "day_of_week",        # 0-4 (Mon-Fri), normalised to [-1, 1]
    "month_of_year",      # 1-12, normalised to [-1, 1]
    "days_to_month_end",  # Trading days to month end, normalised
]

ALL_FEATURE_NAMES = (
    MOMENTUM_FEATURES + VOLATILITY_FEATURES + VOLUME_FEATURES +
    TREND_FEATURES + MEAN_REVERSION_FEATURES + CROSS_STOCK_FEATURES +
    ROLLING_STAT_FEATURES + CALENDAR_FEATURES
)


class AlphaFeatureEngine:
    """
    Computes 50+ alpha features for the LightGBM model.

    Usage
    -----
    engine = AlphaFeatureEngine(config)
    panel  = engine.build_features(data)  # data = {ticker: DataFrame}
    X, y   = engine.build_training_set(panel, forward_horizon=5, purge_gap=5)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.alpha_cfg = config.get("alpha", {})
        self.tickers = config["market"]["tickers"]

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return the ordered list of all alpha feature columns."""
        return list(ALL_FEATURE_NAMES)

    def get_n_features(self) -> int:
        return len(ALL_FEATURE_NAMES)

    # ── Main API ─────────────────────────────────────────────────────────────

    def build_features(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all alpha features for each ticker.

        Parameters
        ----------
        data : dict of {ticker: DataFrame} with OHLCV columns

        Returns
        -------
        dict of {ticker: DataFrame} with all alpha feature columns appended
        """
        # Step 1: Compute per-stock features
        enriched = {}
        for ticker, df in data.items():
            feat = self._compute_single_stock_features(df.copy())
            enriched[ticker] = feat

        # Step 2: Compute cross-stock features (needs full panel)
        enriched = self._compute_cross_stock_features(enriched)

        # Step 3: Compute calendar features
        for ticker, df in enriched.items():
            enriched[ticker] = self._compute_calendar_features(df)

        # Step 4: Drop warmup NaN rows and validate
        for ticker in list(enriched.keys()):
            df = enriched[ticker]
            n_before = len(df)
            df = df.dropna(subset=ALL_FEATURE_NAMES)
            enriched[ticker] = df
            if len(df) < 50:
                logger.warning(
                    "%s: Only %d rows after feature computation (was %d) — skipping.",
                    ticker, len(df), n_before,
                )
                del enriched[ticker]

        logger.info(
            "Alpha features built: %d tickers × %d features, "
            "min_rows=%d, max_rows=%d",
            len(enriched), len(ALL_FEATURE_NAMES),
            min(len(df) for df in enriched.values()) if enriched else 0,
            max(len(df) for df in enriched.values()) if enriched else 0,
        )
        return enriched

    def build_training_set(
        self,
        data: Dict[str, pd.DataFrame],
        forward_horizon: int = 5,
        purge_gap: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build (X, y) for LightGBM training with forward return labels.

        Parameters
        ----------
        data             : dict of {ticker: DataFrame} with alpha features
        forward_horizon  : number of days ahead for return label
        purge_gap        : embargo gap (days) between train/test to prevent leakage

        Returns
        -------
        X : DataFrame of shape (n_samples, n_features) with ticker/date index
        y : Series of forward returns (log)
        """
        frames = []
        for ticker, df in data.items():
            if "Close" not in df.columns:
                continue
            sub = df[ALL_FEATURE_NAMES].copy()
            # Forward return label (log return over horizon)
            sub["forward_return"] = np.log(
                df["Close"].shift(-forward_horizon) / df["Close"]
            )
            sub["ticker"] = ticker
            sub["date"] = df.index
            frames.append(sub)

        if not frames:
            raise ValueError("No data available for training set construction.")

        panel = pd.concat(frames, ignore_index=True)

        # Drop rows where forward return is NaN (end of series)
        panel = panel.dropna(subset=["forward_return"] + ALL_FEATURE_NAMES)
        # Preserve the exact sample identity. Downstream purged CV must use the
        # dates of rows that survived feature normalization/dropna rather than
        # reconstructing an approximate, misaligned date vector.
        panel = panel.set_index(["date", "ticker"]).sort_index()

        X = panel[ALL_FEATURE_NAMES]
        y = panel["forward_return"]

        logger.info(
            "Training set: %d samples, %d features, "
            "y_mean=%.4f, y_std=%.4f",
            len(X), X.shape[1], y.mean(), y.std(),
        )
        return X, y

    def build_purged_cv_splits(
        self,
        dates: pd.Series,
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged time-series cross-validation splits.

        Each fold uses an expanding window for training and a fixed-size
        test window. A purge gap of `purge_gap` days is enforced between
        train and test to prevent information leakage from overlapping
        forward return labels.

        Parameters
        ----------
        dates     : Series of datetime values aligned with X/y
        n_splits  : Number of CV folds
        purge_gap : Number of days to embargo between train and test

        Returns
        -------
        List of (train_indices, test_indices) tuples
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        test_size = n_dates // (n_splits + 1)
        min_train_size = test_size * 2  # Minimum 2× test size for training

        splits = []
        for i in range(n_splits):
            test_start_idx = min_train_size + i * test_size
            test_end_idx = min(test_start_idx + test_size, n_dates)

            if test_end_idx > n_dates:
                break

            test_dates = set(unique_dates[test_start_idx:test_end_idx])

            # Purge: remove `purge_gap` dates before test start
            purge_start = max(0, test_start_idx - purge_gap)
            purge_dates = set(unique_dates[purge_start:test_start_idx])

            train_dates = set(unique_dates[:test_start_idx]) - purge_dates

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        logger.info(
            "Purged CV: %d splits, purge_gap=%d days", len(splits), purge_gap
        )
        return splits

    # ── Per-stock features ───────────────────────────────────────────────────

    def _compute_single_stock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all single-stock features on a raw OHLCV DataFrame."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        log_ret = np.log(close / close.shift(1))

        # ── 1. Momentum & Returns ────────────────────────────────────────────
        df["ret_1d"] = log_ret
        df["ret_2d"] = np.log(close / close.shift(2))
        df["ret_5d"] = np.log(close / close.shift(5))
        df["ret_10d"] = np.log(close / close.shift(10))
        df["ret_20d"] = np.log(close / close.shift(20))
        df["ret_60d"] = np.log(close / close.shift(60))

        # 12-1 momentum: return over months 2-12 (skip most recent month)
        df["mom_12_1"] = np.log(close.shift(21) / close.shift(252))

        # Rate of change
        df["roc_5"] = (close - close.shift(5)) / close.shift(5)
        df["roc_20"] = (close - close.shift(20)) / close.shift(20)

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high, low, close, lbp=14
        ).williams_r() / 100.0  # Normalise from [-100, 0] to [-1, 0]

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch() / 100.0
        df["stoch_d"] = stoch.stoch_signal() / 100.0

        # ── 2. Volatility & Risk ─────────────────────────────────────────────
        df["vol_5d"] = log_ret.rolling(5).std()
        df["vol_10d"] = log_ret.rolling(10).std()
        df["vol_20d"] = log_ret.rolling(20).std()
        df["vol_60d"] = log_ret.rolling(60).std()

        # Vol regime ratio: short-term / long-term
        df["vol_ratio_5_20"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)

        # ATR
        atr_obj = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["atr_14"] = atr_obj.average_true_range()
        df["atr_ratio"] = df["atr_14"] / close  # Normalised by price

        # Parkinson high-low volatility estimator
        log_hl = np.log(high / low)
        df["parkinson_vol"] = log_hl.rolling(20).apply(
            lambda x: np.sqrt((1 / (4 * np.log(2))) * np.mean(x**2)),
            raw=True,
        )

        # ── 3. Volume Anomalies ──────────────────────────────────────────────
        vol_mean_20 = volume.rolling(20).mean()
        vol_std_20 = volume.rolling(20).std().replace(0, 1)
        vol_mean_60 = volume.rolling(60).mean()
        vol_std_60 = volume.rolling(60).std().replace(0, 1)

        df["volume_z_20"] = (volume - vol_mean_20) / vol_std_20
        df["volume_z_60"] = (volume - vol_mean_60) / vol_std_60

        # OBV slope (On-Balance Volume 5-day linear trend)
        obv = (np.sign(log_ret) * volume).cumsum()
        df["obv_slope"] = obv.rolling(5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 5 else 0,
            raw=True,
        ) / vol_mean_20.replace(0, 1)  # Normalise by avg volume

        # VWAP proxy distance (rolling VWAP over 20 periods)
        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).rolling(20).sum()
        cum_vol = volume.rolling(20).sum().replace(0, 1)
        vwap = cum_tp_vol / cum_vol
        df["vwap_dist"] = (close - vwap) / vwap

        # Volume breakout (binary: volume > 2× 20d average)
        df["volume_breakout"] = (volume > 2 * vol_mean_20).astype(float)

        # Volume-price correlation (rolling 20d)
        abs_ret = log_ret.abs()
        df["volume_price_corr"] = volume.rolling(20).corr(abs_ret)

        # ── 4. Trend Indicators ──────────────────────────────────────────────
        for period in [9, 21, 50, 200]:
            ema = ta.trend.EMAIndicator(close, window=period).ema_indicator()
            df[f"ema_dist_{period}"] = (close - ema) / ema

        # ADX (Average Directional Index)
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx_14"] = adx_obj.adx() / 100.0  # Normalise to [0, 1]

        # CCI (Commodity Channel Index)
        df["cci_20"] = ta.trend.CCIIndicator(
            high, low, close, window=20
        ).cci() / 200.0  # Normalise: CCI typically ranges ±200

        # MACD histogram
        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd_hist"] = macd_obj.macd_diff() / close  # Normalised

        # Trend strength: ADX × direction
        ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        df["trend_strength"] = (df["adx_14"]) * np.sign(close - ema_50)

        # ── 5. Mean Reversion ────────────────────────────────────────────────
        df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0
        df["rsi_5"] = ta.momentum.RSIIndicator(close, window=5).rsi() / 100.0

        bb_obj = ta.volatility.BollingerBands(close, window=20, window_dev=2.0)
        df["bb_pct"] = bb_obj.bollinger_pband()
        df["bb_width"] = (bb_obj.bollinger_hband() - bb_obj.bollinger_lband()) / close

        # Distance from 52-week high
        high_52w = close.rolling(252, min_periods=60).max()
        df["dist_from_52w_high"] = (close - high_52w) / high_52w

        # ── 6. Rolling Statistical ───────────────────────────────────────────
        df["skewness_20"] = log_ret.rolling(20).skew()
        df["kurtosis_20"] = log_ret.rolling(20).kurt()
        df["skewness_60"] = log_ret.rolling(60).skew()
        df["kurtosis_60"] = log_ret.rolling(60).kurt()

        # Lag-1 autocorrelation (20-day rolling)
        df["autocorr_1"] = log_ret.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else 0,
            raw=False,
        )

        # Rolling 20-day max drawdown
        cummax_20 = close.rolling(20).max()
        df["max_drawdown_20"] = (close - cummax_20) / cummax_20

        return df

    # ── Cross-stock features (needs full panel) ──────────────────────────────

    def _compute_cross_stock_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute cross-sectional features that compare each stock to the panel.
        """
        tickers = list(data.keys())
        if len(tickers) < 2:
            # Can't compute relative features with < 2 stocks
            for ticker in tickers:
                for col in CROSS_STOCK_FEATURES:
                    data[ticker][col] = 0.0
            return data

        # Collect per-date return and vol panels
        # Find common dates
        common_dates = None
        for df in data.values():
            idx = set(df.index)
            common_dates = idx if common_dates is None else common_dates & idx
        common_dates = sorted(common_dates)

        # Build panels: (n_dates, n_tickers)
        ret_panel = pd.DataFrame(
            {t: data[t].reindex(common_dates)["ret_20d"] for t in tickers},
            index=common_dates,
        )
        ret_1d_panel = pd.DataFrame(
            {t: data[t].reindex(common_dates)["ret_1d"] for t in tickers},
            index=common_dates,
        )
        vol_panel = pd.DataFrame(
            {t: data[t].reindex(common_dates)["vol_20d"] for t in tickers},
            index=common_dates,
        )

        # Cross-sectional statistics per date
        cs_ret_mean = ret_1d_panel.mean(axis=1)
        cs_ret_std = ret_1d_panel.std(axis=1).replace(0, 1)
        ret_ranks = ret_1d_panel.rank(axis=1, pct=True)
        vol_ranks = vol_panel.rank(axis=1, pct=True)
        mom_ranks = ret_panel.rank(axis=1, pct=True)

        for ticker in tickers:
            df = data[ticker]
            common_mask = df.index.isin(common_dates)

            # Return rank (percentile across stocks at this date). Ranking a
            # ticker down its entire time series leaks future observations.
            df.loc[common_mask, "ret_rank"] = ret_ranks[ticker].reindex(
                df.index[common_mask]
            ).values

            # Volatility rank across stocks at this date.
            df.loc[common_mask, "vol_rank"] = vol_ranks[ticker].reindex(
                df.index[common_mask]
            ).values

            # Momentum rank (20d return rank across stocks at each date)
            df.loc[common_mask, "mom_rank"] = mom_ranks[ticker].reindex(
                df.index[common_mask]
            ).values

            # Return vs cross-sectional mean
            df.loc[common_mask, "ret_vs_mean"] = (
                (ret_1d_panel[ticker] - cs_ret_mean) / cs_ret_std
            ).reindex(df.index[common_mask]).values

            # Sector-relative momentum (vs equal-weight basket)
            basket_ret = ret_panel.mean(axis=1)
            df.loc[common_mask, "sector_rel_mom"] = (
                ret_panel[ticker] - basket_ret
            ).reindex(df.index[common_mask]).values

            # Dispersion contribution
            df.loc[common_mask, "dispersion_contrib"] = (
                (ret_1d_panel[ticker] - cs_ret_mean).abs() / cs_ret_std
            ).reindex(df.index[common_mask]).values

            # Fill non-common dates with 0
            for col in CROSS_STOCK_FEATURES:
                df[col] = df[col].fillna(0.0)

            data[ticker] = df

        return data

    # ── Calendar features ────────────────────────────────────────────────────

    def _compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute calendar/structural features from the date index."""
        dates = pd.DatetimeIndex(df.index)

        # Day of week: 0 (Mon) to 4 (Fri) → normalised to [-1, 1]
        df["day_of_week"] = (dates.dayofweek - 2) / 2.0

        # Month of year: 1-12 → normalised to [-1, 1]
        df["month_of_year"] = (dates.month - 6.5) / 5.5

        # Days to month end (approximate using calendar)
        month_end = dates + pd.offsets.MonthEnd(0)
        df["days_to_month_end"] = (month_end - dates).days
        max_dtme = df["days_to_month_end"].max()
        if max_dtme > 0:
            df["days_to_month_end"] = df["days_to_month_end"] / max_dtme
        else:
            df["days_to_month_end"] = 0.0

        return df

    # ── Z-score normalisation ────────────────────────────────────────────────

    @staticmethod
    def zscore_features(
        data: Dict[str, pd.DataFrame],
        window: int = 252,
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply rolling z-score normalisation to all alpha features.
        Uses only past data — no look-ahead bias.
        """
        for ticker, df in data.items():
            feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
            means = df[feature_cols].rolling(window, min_periods=20).mean()
            stds = df[feature_cols].rolling(window, min_periods=20).std().replace(0, 1)
            df[feature_cols] = (df[feature_cols] - means) / stds
            # Clip extreme values
            df[feature_cols] = df[feature_cols].clip(-5, 5)
            data[ticker] = df
        return data

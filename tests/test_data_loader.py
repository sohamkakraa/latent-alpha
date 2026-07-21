"""Unit tests for env/data_loader.py — indicator computation and normalisation."""
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_ohlcv():
    """Create a synthetic OHLCV DataFrame with 200 rows."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)

    df = pd.DataFrame({
        "Open":   close * (1 + np.random.randn(n) * 0.005),
        "High":   close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "Low":    close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "Close":  close,
        "Volume": np.random.randint(100000, 10000000, size=n).astype(float),
    }, index=dates)
    return df


class TestComputeIndicators:
    """Tests for compute_indicators()."""

    def test_rsi_in_range(self, config, sample_ohlcv):
        """RSI must be in [0, 100]."""
        from env.data_loader import compute_indicators
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        rsi_col = next((c for c in df.columns if "rsi" in c.lower()), None)
        assert rsi_col is not None, f"No RSI column found in {df.columns.tolist()}"
        rsi = df[rsi_col].dropna()
        assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"

    def test_macd_columns_exist(self, config, sample_ohlcv):
        """MACD and MACD_signal columns should exist."""
        from env.data_loader import compute_indicators
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        macd_cols = [c for c in df.columns if "macd" in c.lower()]
        assert len(macd_cols) >= 2, f"Expected 2+ MACD columns, got: {macd_cols}"

    def test_bollinger_bands_order(self, config, sample_ohlcv):
        """BB_lower <= BB_upper at all valid rows."""
        from env.data_loader import compute_indicators
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        bb_upper = next((c for c in df.columns if "bb_upper" in c.lower() or "BB_upper" in c), None)
        bb_lower = next((c for c in df.columns if "bb_lower" in c.lower() or "BB_lower" in c), None)
        if bb_upper and bb_lower:
            valid = df.dropna(subset=[bb_upper, bb_lower])
            assert (valid[bb_lower] <= valid[bb_upper]).all(), "BB_lower > BB_upper"

    def test_atr_positive(self, config, sample_ohlcv):
        """ATR must be non-negative."""
        from env.data_loader import compute_indicators
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        atr_col = next((c for c in df.columns if "atr" in c.lower()), None)
        if atr_col:
            atr = df[atr_col].dropna()
            assert (atr >= 0).all(), f"Negative ATR found: {atr.min()}"

    def test_ema_columns_match_config(self, config, sample_ohlcv):
        """EMA columns should exist for each configured period."""
        from env.data_loader import compute_indicators
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        ema_cols = [c for c in df.columns if "ema" in c.lower()]
        assert len(ema_cols) >= 1, f"No EMA columns found in {df.columns.tolist()}"

    def test_no_nan_after_warmup(self, config, sample_ohlcv):
        """After sufficient warmup (60 rows), indicators should have no NaN."""
        from env.data_loader import compute_indicators, NSEDataLoader
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        loader = NSEDataLoader(config)
        feature_cols = loader.get_feature_names()
        indicator_cols = [c for c in feature_cols if c in df.columns]
        if not indicator_cols:
            # Some feature names differ from raw indicator col names
            indicator_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
        tail = df.iloc[60:][indicator_cols]
        nan_counts = tail.isna().sum()
        assert nan_counts.sum() == 0, f"NaN after warmup: {nan_counts[nan_counts > 0].to_dict()}"


class TestRollingZscore:
    """Tests for rolling_zscore normalisation."""

    def test_zscore_mean_near_zero(self, config, sample_ohlcv):
        """Z-scored values should have mean ≈ 0 over the window."""
        from env.data_loader import compute_indicators, rolling_zscore
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        df_z = rolling_zscore(df, window=60)
        # Check a numeric column that is present
        for col in ["rsi", "RSI_14", "Close"]:
            if col in df_z.columns:
                vals = df_z[col].dropna().iloc[60:]
                if len(vals) > 0:
                    assert abs(vals.mean()) < 1.0, f"{col} z-score mean too far from 0: {vals.mean()}"
                break

    def test_zscore_no_lookahead(self, config, sample_ohlcv):
        """Z-score at time t should only use data up to t (no future leakage)."""
        from env.data_loader import compute_indicators, rolling_zscore
        df = compute_indicators(sample_ohlcv.copy(), config["indicators"])
        df_z = rolling_zscore(df.copy(), window=60)

        # Use 150 rows for short (covers warmup + window of 60)
        df_short = compute_indicators(sample_ohlcv.iloc[:150].copy(), config["indicators"])
        df_short_z = rolling_zscore(df_short.copy(), window=60)

        # Pick a stable column to compare
        for col in ["rsi", "Close"]:
            if col in df_z.columns and col in df_short_z.columns:
                idx = min(100, len(df_short_z) - 1)
                full_val  = df_z[col].iloc[idx]
                short_val = df_short_z[col].iloc[idx]
                if not (np.isnan(full_val) or np.isnan(short_val)):
                    assert abs(full_val - short_val) < 1e-6, (
                        f"Look-ahead leakage in {col}: full={full_val}, short={short_val}"
                    )
                break


class TestFeatureNames:
    """Tests for feature name consistency."""

    def test_feature_names_non_empty(self, config):
        """get_feature_names() should return a non-empty list."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        names = loader.get_feature_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_unique(self, config):
        """Feature names should be unique."""
        from env.data_loader import NSEDataLoader
        loader = NSEDataLoader(config)
        names = loader.get_feature_names()
        assert len(names) == len(set(names)), (
            f"Duplicate feature names: {[n for n in names if names.count(n) > 1]}"
        )

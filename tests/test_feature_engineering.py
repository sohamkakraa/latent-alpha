"""Unit tests for alpha/feature_engineering.py — 54 features across 8 groups."""
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_multi_stock_data():
    """Create synthetic OHLCV data for 3 stocks (enough for 12-month momentum features)."""
    np.random.seed(42)
    n = 600  # 600 business days (~2.4 years) — covers mom_12_1 warmup
    dates = pd.bdate_range("2018-01-01", periods=n)

    data = {}
    for ticker in ["STOCK_A", "STOCK_B", "STOCK_C"]:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        df = pd.DataFrame({
            "Open":   close * (1 + np.random.randn(n) * 0.005),
            "High":   close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low":    close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close":  close,
            "Volume": np.random.randint(100000, 10000000, size=n).astype(float),
        }, index=dates)
        data[ticker] = df
    return data


class TestAlphaFeatureEngine:
    """Tests for AlphaFeatureEngine.build_features()."""

    def test_build_features_returns_dict(self, config, sample_multi_stock_data):
        """build_features() should return a dict of DataFrames."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)

        assert isinstance(result, dict)
        assert len(result) == len(sample_multi_stock_data)
        for ticker, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_feature_count(self, config, sample_multi_stock_data):
        """Should produce 40+ features per stock."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        engine.build_features(sample_multi_stock_data)

        feature_names = engine.get_feature_names()
        assert len(feature_names) >= 40, f"Only {len(feature_names)} features (expected 40+)"

    def test_feature_names_unique(self, config, sample_multi_stock_data):
        """Feature names should be unique."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        engine.build_features(sample_multi_stock_data)

        names = engine.get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_momentum_features_present(self, config, sample_multi_stock_data):
        """Momentum group should produce ret_1d, ret_5d, ret_20d, etc."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)

        ticker = list(result.keys())[0]
        df = result[ticker]
        momentum_cols = [c for c in df.columns if "ret_" in c.lower() or "mom" in c.lower()]
        assert len(momentum_cols) >= 3, f"Expected 3+ momentum features, got {len(momentum_cols)}"

    def test_cross_stock_features_present(self, config, sample_multi_stock_data):
        """Cross-stock group should produce rank and relative features."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)

        ticker = list(result.keys())[0]
        df = result[ticker]
        cross_cols = [
            c for c in df.columns
            if "rank" in c.lower() or "rel" in c.lower() or "dispersion" in c.lower()
        ]
        assert len(cross_cols) >= 2, f"Expected 2+ cross-stock features, got {len(cross_cols)}"

    def test_no_nan_after_warmup(self, config, sample_multi_stock_data):
        """After 120 days warmup, features should have no NaN."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)

        for ticker, df in result.items():
            feature_names = engine.get_feature_names()
            feat_cols = [c for c in feature_names if c in df.columns]
            tail = df.iloc[120:][feat_cols]
            nan_pct = tail.isna().mean()
            bad_cols = nan_pct[nan_pct > 0.05]
            assert len(bad_cols) == 0, (
                f"{ticker}: columns with >5% NaN after warmup: {bad_cols.to_dict()}"
            )

    def test_no_inf_values(self, config, sample_multi_stock_data):
        """Features should not contain inf values."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)

        for ticker, df in result.items():
            numeric_df = df.select_dtypes(include=[np.number])
            inf_count = np.isinf(numeric_df.values).sum()
            assert inf_count == 0, f"{ticker}: {inf_count} inf values found"

    def test_zscore_features(self, config, sample_multi_stock_data):
        """zscore_features() should normalise without introducing extreme values."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)

        for ticker, df in normed.items():
            feature_names = engine.get_feature_names()
            feat_cols = [c for c in feature_names if c in df.columns]
            tail = df.iloc[120:][feat_cols]
            max_abs = tail.abs().max().max()
            assert max_abs < 100, f"{ticker}: z-scored feature has extreme value {max_abs}"


class TestBuildTrainingSet:
    """Tests for building training labels."""

    def test_training_set_shapes(self, config, sample_multi_stock_data):
        """X and y should have matching first dimension."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)

        X, y = engine.build_training_set(normed, forward_horizon=10, purge_gap=5)
        assert X.shape[0] == y.shape[0], f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})"
        assert X.shape[0] > 0, "Training set is empty"
        assert X.shape[1] > 0, "No features in training set"

    def test_labels_are_returns(self, config, sample_multi_stock_data):
        """y (forward returns) should be centered near 0."""
        from alpha.feature_engineering import AlphaFeatureEngine
        engine = AlphaFeatureEngine(config)
        result = engine.build_features(sample_multi_stock_data)
        normed = engine.zscore_features(result)

        X, y = engine.build_training_set(normed, forward_horizon=10, purge_gap=5)
        assert abs(y.mean()) < 0.1, f"Labels mean too far from 0: {y.mean()}"
        assert y.std() > 0, "Labels have zero variance"

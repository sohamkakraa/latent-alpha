"""Tests for meta-model activation and signal alignment (P0 Prompt 2)."""
import numpy as np
import pytest


def test_gnn_forward_horizon_aligned():
    """All channels must use the same forward_horizon."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    alpha_horizon   = config["alpha"]["forward_horizon"]
    patchtst_horizon = config["patchtst"]["forward_horizon"]
    gnn_horizon     = config["gnn"]["forward_horizon"]

    assert alpha_horizon == patchtst_horizon == gnn_horizon, (
        f"Horizons misaligned: alpha={alpha_horizon}, "
        f"patchtst={patchtst_horizon}, gnn={gnn_horizon}"
    )


def test_meta_model_config_exists():
    """Verify meta_model config section exists with correct fields."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    assert "meta_model" in config
    mc = config["meta_model"]
    assert mc["enabled"] == True
    assert mc["obs_dim"] == 20
    assert mc["ridge_alpha"] > 0
    assert mc["min_train_samples"] > 0


def test_ridge_meta_model_train_and_combine():
    """Verify RidgeMetaModel can train and combine predictions."""
    from alpha.meta_model import RidgeMetaModel, ChannelPrediction

    np.random.seed(0)
    meta = RidgeMetaModel(n_stocks=10, ridge_alpha=0.01)
    meta.register_channel("lgbm_alpha")
    meta.register_channel("patchtst")

    n_samples = 200
    signal_a = np.random.randn(n_samples)
    signal_b = np.random.randn(n_samples)
    preds = {
        "lgbm_alpha": signal_a,
        "patchtst":   signal_b,
    }
    # Clean linear combination so Ridge should recover high R²
    actuals = 0.6 * signal_a + 0.4 * signal_b + np.random.randn(n_samples) * 0.01

    result = meta.train(preds, actuals)
    assert "r2" in result
    assert "weights" in result
    assert result["r2"] > 0.5, f"Expected R²>0.5, got {result['r2']:.4f}"

    cp = [
        ChannelPrediction(
            name="lgbm_alpha",
            direction=np.random.randn(10).astype(np.float32),
            conviction=np.abs(np.random.randn(10)).astype(np.float32) * 0.5,
        ),
        ChannelPrediction(
            name="patchtst",
            direction=np.random.randn(10).astype(np.float32),
            conviction=np.abs(np.random.randn(10)).astype(np.float32) * 0.5,
        ),
    ]
    direction, conviction = meta.combine(cp)
    assert direction.shape == (10,)
    assert conviction.shape == (10,)


def test_combined_alpha_fn_returns_alpha_signal():
    """Verify AlphaSignal interface is correct."""
    from alpha.lightgbm_alpha import AlphaSignal

    sig = AlphaSignal(
        direction=np.random.randn(10).astype(np.float32),
        conviction=np.abs(np.random.randn(10)).astype(np.float32),
        raw_pred=np.random.randn(10).astype(np.float32),
    )
    obs = sig.to_obs_vector()
    assert obs.shape == (20,)  # 10 direction + 10 conviction


def test_obs_dim_with_meta_model():
    """Meta-model obs_dim should be smaller than all separate channels combined."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    meta_obs_dim = config["meta_model"]["obs_dim"]  # 20

    separate_obs_dim = (
        config["alpha"]["obs_dim"]
        + config["patchtst"]["obs_dim"]
        + config["gnn"]["obs_dim"]
        + config["vae_regime"]["obs_dim"]
        + config["llm"]["debate"]["obs_dim"]
    )

    assert meta_obs_dim < separate_obs_dim, (
        f"Meta-model obs ({meta_obs_dim}) should be smaller than "
        f"separate channels ({separate_obs_dim})"
    )


def test_ridge_meta_model_kwargs_constructor():
    """RidgeMetaModel should accept n_stocks and ridge_alpha as kwargs."""
    from alpha.meta_model import RidgeMetaModel

    meta = RidgeMetaModel(n_stocks=10, ridge_alpha=2.0)
    assert meta.n_stocks == 10
    assert meta.ridge_alpha == 2.0

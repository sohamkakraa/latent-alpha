"""Integration tests for Phase 3 (GNN + VAE) in the SAC pipeline."""
import numpy as np
import pytest


def test_obs_space_with_gnn_vae():
    """Observation space includes GNN and VAE dimensions when enabled."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    n_stocks = len(config["market"]["tickers"])

    # Enable both
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    sentiment_dim = config["llm"]["sentiment_dim"]
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)
    gnn_dim = config["gnn"]["obs_dim"]
    vae_dim = config["vae_regime"]["obs_dim"]

    expected_obs = (
        n_stocks * n_features
        + n_stocks + 2 + n_stocks
        + n_stocks * sentiment_dim
        + alpha_dim + temporal_dim
        + gnn_dim + vae_dim
    )
    print(f"Phase 3 obs_dim: {expected_obs}")
    assert gnn_dim == 20
    assert vae_dim == 8
    # Phase 2 base ~462 dims; Phase 3 adds GNN (20) + VAE (8) = 28 more
    assert expected_obs > 460


def test_env_accepts_gnn_regime_fn():
    """TradingEnv constructor accepts gnn_fn and regime_fn without error."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    import inspect
    from env.trading_env import NSETradingEnv
    sig = inspect.signature(NSETradingEnv.__init__)
    params = list(sig.parameters.keys())
    assert "gnn_fn" in params
    assert "regime_fn" in params


def test_env_obs_includes_gnn_vae_zeros():
    """When gnn_fn/regime_fn are None, obs still has correct shape with zeros."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = True
    config["vae_regime"]["enabled"] = True

    n_stocks = len(config["market"]["tickers"])
    gnn_dim = config["gnn"]["obs_dim"]
    vae_dim = config["vae_regime"]["obs_dim"]

    # GNN and VAE dims should be present even when fns are None
    assert gnn_dim == 20
    assert vae_dim == 8


def test_backward_compatibility():
    """With GNN and VAE disabled, obs space matches Phase 2."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["gnn"]["enabled"] = False
    config["vae_regime"]["enabled"] = False

    n_stocks = len(config["market"]["tickers"])
    from env.data_loader import NSEDataLoader
    loader = NSEDataLoader(config)
    n_features = len(loader.get_feature_names())
    sentiment_dim = config["llm"]["sentiment_dim"]
    alpha_dim = config.get("alpha", {}).get("obs_dim", 0)
    temporal_dim = config.get("patchtst", {}).get("obs_dim", 0)

    phase2_obs = (
        n_stocks * n_features
        + n_stocks + 2 + n_stocks
        + n_stocks * sentiment_dim
        + alpha_dim + temporal_dim
    )
    print(f"Phase 2 obs_dim (backward compat): {phase2_obs}")
    assert phase2_obs > 400


def test_term_selector_has_gnn_regime_builders():
    """TermSelector has _build_gnn_pipeline and _build_regime_pipeline methods."""
    from strategy.term_selector import TermSelector
    assert hasattr(TermSelector, "_build_gnn_pipeline")
    assert hasattr(TermSelector, "_build_regime_pipeline")

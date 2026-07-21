"""Tests for multi-horizon portfolio orchestrator (P1 Prompt 8)."""
import numpy as np
import pytest


def test_equal_weight_combination():
    """Equal weight should average across horizons."""
    from strategy.multi_horizon import MultiHorizonOrchestrator

    weights = {
        "short":  np.array([0.1, 0.2, -0.1] + [0.0] * 7),
        "medium": np.array([0.2, 0.0,  0.1] + [0.0] * 7),
        "long":   np.array([0.0, 0.1,  0.0] + [0.0] * 7),
    }

    combined = MultiHorizonOrchestrator._equal_weight(weights)
    expected = np.array([0.1, 0.1, 0.0] + [0.0] * 7)
    np.testing.assert_allclose(combined, expected, atol=1e-6)


def test_risk_parity_favours_low_vol():
    """Risk parity should give more weight to lower-volatility horizons."""
    from strategy.multi_horizon import MultiHorizonOrchestrator

    weights = {
        "medium": np.array([0.2] * 10),
        "long":   np.array([0.1] * 10),
    }
    vols = {"medium": 0.2, "long": 0.1}  # Long is less volatile

    combined = MultiHorizonOrchestrator._risk_parity(weights, vols)
    # Long (lower vol) gets more weight, pulls combined toward 0.1
    assert np.mean(combined) < 0.2, "Should be pulled toward long's lower weights"
    assert np.mean(combined) > 0.1, "Should not be fully at long's weights"


def test_combination_preserves_leverage_constraint():
    """Combined weights should not exceed 1.5× leverage."""
    from strategy.multi_horizon import MultiHorizonOrchestrator

    weights = {
        "short":  np.array([0.3] * 10),
        "medium": np.array([0.2] * 10),
    }

    combined = MultiHorizonOrchestrator._equal_weight(weights)
    total_exposure = np.sum(np.abs(combined))
    assert total_exposure <= 1.5, f"Combined exposure {total_exposure} exceeds 1.5× leverage"


def test_sharpe_weighted_all_negative():
    """When all Sharpes are negative, should fall back to equal weight."""
    from strategy.multi_horizon import MultiHorizonOrchestrator

    weights = {
        "medium": np.array([0.2] * 10),
        "long":   np.array([0.1] * 10),
    }
    sharpes = {"medium": -0.5, "long": -0.3}

    combined = MultiHorizonOrchestrator._sharpe_weighted(weights, sharpes)
    # Falls back to equal weight
    expected = MultiHorizonOrchestrator._equal_weight(weights)
    np.testing.assert_allclose(combined, expected, atol=1e-6)


def test_multi_horizon_config_exists():
    """Config should have multi_horizon section with enabled: false default."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    assert "multi_horizon" in config
    mh = config["multi_horizon"]
    assert "enabled" in mh
    assert mh["enabled"] == False   # Disabled by default
    assert "horizons" in mh
    assert "combination" in mh


def test_orchestrator_init():
    """MultiHorizonOrchestrator should initialise without error."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    from strategy.multi_horizon import MultiHorizonOrchestrator
    orc = MultiHorizonOrchestrator(config, risk_profile="balanced", horizons=["medium"])
    assert "medium" in orc.horizons
    assert orc.combination in ("equal_weight", "risk_parity", "sharpe_weighted")


def test_combine_weights_interface():
    """combine_weights() public method should work correctly."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    from strategy.multi_horizon import MultiHorizonOrchestrator
    orc = MultiHorizonOrchestrator(config, risk_profile="balanced")

    weights = {
        "medium": np.array([0.1] * 10, dtype=np.float32),
        "long":   np.array([0.2] * 10, dtype=np.float32),
    }
    combined = orc.combine_weights(weights, strategy="equal_weight")
    assert combined.shape == (10,)
    np.testing.assert_allclose(combined, np.array([0.15] * 10), atol=1e-6)

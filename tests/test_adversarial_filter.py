"""Tests for adversarial conviction filter."""
import numpy as np
import pytest


def test_filter_disabled_passthrough():
    from alpha.adversarial_filter import AdversarialFilter
    from alpha.lightgbm_alpha import AlphaSignal
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["alpha"]["adversarial_filter"]["enabled"] = False

    filt = AdversarialFilter(config)
    signal = AlphaSignal(
        direction=np.array([0.8, -0.5, 0.3]),
        conviction=np.array([0.9, 0.7, 0.4]),
        raw_pred=np.array([0.02, -0.01, 0.005]),
    )
    result = filt.filter(signal, date="2024-01-15")
    np.testing.assert_array_equal(result.direction, signal.direction)
    np.testing.assert_array_equal(result.conviction, signal.conviction)


def test_filter_only_challenges_high_conviction():
    from alpha.adversarial_filter import _select_stocks_to_challenge
    convictions = np.array([0.9, 0.5, 0.3, 0.8, 0.6])
    threshold = 0.7
    indices = _select_stocks_to_challenge(convictions, threshold)
    assert 0 in indices
    assert 3 in indices
    assert 1 not in indices
    assert 2 not in indices
    assert 4 not in indices


def test_filter_never_flips_direction():
    from alpha.adversarial_filter import AdversarialFilter
    from alpha.lightgbm_alpha import AlphaSignal
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["alpha"]["adversarial_filter"]["enabled"] = True
    config["alpha"]["adversarial_filter"]["conviction_threshold"] = 0.7
    config["alpha"]["adversarial_filter"]["counter_conviction_threshold"] = 0.7
    config["alpha"]["adversarial_filter"]["dampening"] = 0.3

    filt = AdversarialFilter(config)
    signal = AlphaSignal(
        direction=np.array([0.8, -0.5]),
        conviction=np.array([0.9, 0.8]),
        raw_pred=np.array([0.02, -0.01]),
    )

    # Mock the LLM to return high counter-conviction
    filt._mock_counter_convictions = {0: 0.9, 1: 0.9}

    result = filt.filter(signal, date="2024-01-15")
    assert np.sign(result.direction[0]) == np.sign(signal.direction[0])
    assert np.sign(result.direction[1]) == np.sign(signal.direction[1])
    assert result.conviction[0] < signal.conviction[0]
    assert result.conviction[1] < signal.conviction[1]


def test_dampening_math():
    from alpha.adversarial_filter import _apply_dampening
    original_conviction = 0.9
    counter_conviction = 0.8
    dampening = 0.3

    new_conv = _apply_dampening(original_conviction, counter_conviction, dampening)
    expected = 0.9 * (1 - 0.3 * 0.8)
    assert abs(new_conv - expected) < 1e-6
    assert new_conv > 0
    assert new_conv <= original_conviction


def test_filter_cache():
    from alpha.adversarial_filter import AdversarialFilter
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    filt = AdversarialFilter(config)
    filt._cache[("2024-01-15", 0)] = 0.6
    assert filt._get_cached("2024-01-15", 0) == 0.6
    assert filt._get_cached("2024-01-15", 1) is None

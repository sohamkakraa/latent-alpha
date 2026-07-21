"""Tests for regime-aware adaptive entropy scaling (P1 Prompt 5)."""
import numpy as np
import pytest


def test_regime_uncertainty_uniform():
    """Uniform regime probs should give uncertainty = 1.0."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    u = compute_regime_uncertainty(probs)
    assert u == pytest.approx(1.0, abs=0.01)


def test_regime_uncertainty_certain():
    """One-hot regime probs should give uncertainty ≈ 0."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    u = compute_regime_uncertainty(probs)
    assert u == pytest.approx(0.0, abs=0.01)


def test_regime_uncertainty_in_range():
    """Uncertainty should always be in [0, 1]."""
    from agent.adaptive_entropy import compute_regime_uncertainty
    for _ in range(100):
        probs = np.random.dirichlet(np.ones(4))
        u = compute_regime_uncertainty(probs)
        assert 0.0 <= u <= 1.0, f"Uncertainty out of range: {u}"


def test_scaled_target_more_negative_with_uncertainty():
    """Higher uncertainty should make target_entropy more negative."""
    base = -5.0
    scale = 0.5

    target_certain   = base * (1 + scale * 0.0)   # -5.0
    target_uncertain = base * (1 + scale * 1.0)   # -7.5

    assert target_uncertain < target_certain, (
        "More uncertainty should give more negative target"
    )


def test_callback_initialises():
    """RegimeAwareEntropyCallback should initialise without error."""
    from agent.adaptive_entropy import RegimeAwareEntropyCallback

    def dummy_regime_fn(date):
        from alpha.vae_regime import RegimeSignal
        return RegimeSignal(
            embedding=np.zeros(8, dtype=np.float32),
            regime_probs=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
            regime_label=0,
        )

    cb = RegimeAwareEntropyCallback(
        regime_fn=dummy_regime_fn,
        base_target_entropy=-5.0,
        uncertainty_scale=0.5,
        adjust_freq=1000,
        ent_coef_floor=0.005,
    )
    assert cb.base_target_entropy == -5.0
    assert cb.uncertainty_scale == 0.5
    assert cb.ent_coef_floor == 0.005


def test_adaptive_entropy_retired_in_favor_of_fixed_ent_coef():
    """Adaptive/regime-aware entropy was retired: fixed ent_coef won every
    experiment. The module stays for research, but config must NOT re-enable it.
    """
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sac = config["sac"]
    assert "ent_coef" in sac          # fixed entropy is the active design
    assert "adaptive_entropy" not in sac

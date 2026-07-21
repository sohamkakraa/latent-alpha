"""Tests for the SAC entropy design.

History: auto entropy tuning collapsed in every experiment (ent_coef → ~0.0004
by 500K steps), and the later auto-with-floor / target_entropy_ratio approach
was also dropped. The current, stable design is a FIXED ent_coef. These tests
pin that current reality.
"""
import math
import pytest
import numpy as np


def test_entropy_uses_fixed_ent_coef():
    """Current design: a single fixed ent_coef, no auto-tuning keys."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sac = config["sac"]
    assert "ent_coef" in sac
    assert isinstance(sac["ent_coef"], (int, float))
    assert 0.0 < sac["ent_coef"] <= 1.0
    # The abandoned auto-entropy knobs must not creep back in.
    assert "target_entropy_ratio" not in sac
    assert "ent_coef_init" not in sac


def test_entropy_floor_callback_clamps():
    """EntropyFloorCallback still constructs (legacy safety net; unused with
    a fixed ent_coef but kept for auto-tuning experiments)."""
    from agent.sac_agent import EntropyFloorCallback

    callback = EntropyFloorCallback(ent_coef_floor=0.005, log_every_n_steps=100)
    assert callback.ent_coef_floor == 0.005
    assert callback._clamp_count == 0


def test_sac_agent_reads_fixed_ent_coef():
    """Verify LatentAlphaSACAgent loads the fixed ent_coef from config."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    from agent.sac_agent import LatentAlphaSACAgent
    agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")

    assert hasattr(agent, "entropy_cfg")
    assert agent.entropy_cfg["ent_coef"] == config["sac"]["ent_coef"]

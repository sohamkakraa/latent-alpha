"""Unit tests for agent/sac_agent.py — SAC agent wrapper."""
import pytest
import yaml
import numpy as np


@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class TestLatentAlphaSACAgent:
    """Tests for agent construction and configuration."""

    def test_init_balanced(self, config):
        """Agent should initialise with balanced profile."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")

        assert agent.risk_profile == "balanced"
        assert agent.term == "medium"
        assert agent.model is None

    def test_init_all_profiles(self, config):
        """Agent should initialise with all risk profiles."""
        from agent.sac_agent import LatentAlphaSACAgent
        for profile in ["conservative", "balanced", "aggressive"]:
            agent = LatentAlphaSACAgent(config, risk_profile=profile, term="medium")
            assert agent.risk_profile == profile

    def test_entropy_config_loaded(self, config):
        """Agent should load the fixed ent_coef (auto-tuning was removed)."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")

        assert hasattr(agent, "entropy_cfg")
        assert "ent_coef" in agent.entropy_cfg

    def test_entropy_values_correct(self, config):
        """Fixed ent_coef should match config.yaml."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")

        assert agent.entropy_cfg["ent_coef"] == config["sac"]["ent_coef"]

    def test_model_dir_created(self, config, tmp_path):
        """Model directory should be created on init."""
        cfg = config.copy()
        cfg["paths"] = config.get("paths", {}).copy()
        cfg["paths"]["model_dir"] = str(tmp_path / "models")

        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(cfg, risk_profile="balanced", term="medium")

        assert (tmp_path / "models").exists()


class TestCallbacks:
    """Tests for custom callback classes."""

    def test_activity_check_callback_init(self):
        """ActivityCheckCallback should initialise without error."""
        from agent.sac_agent import ActivityCheckCallback
        cb = ActivityCheckCallback(check_every_n_steps=1000)
        assert cb.check_every == 1000

    def test_training_progress_callback_init(self):
        """TrainingProgressCallback should initialise without error."""
        from agent.sac_agent import TrainingProgressCallback
        cb = TrainingProgressCallback(log_every_n_steps=5000)
        assert cb.log_every == 5000

    def test_entropy_floor_callback_init(self):
        """EntropyFloorCallback should initialise with correct floor."""
        from agent.sac_agent import EntropyFloorCallback
        cb = EntropyFloorCallback(ent_coef_floor=0.005)
        assert cb.ent_coef_floor == 0.005
        assert cb._clamp_count == 0

    def test_entropy_floor_callback_log_interval(self):
        """EntropyFloorCallback log_every should be set correctly."""
        from agent.sac_agent import EntropyFloorCallback
        cb = EntropyFloorCallback(ent_coef_floor=0.005, log_every_n_steps=10_000)
        assert cb.log_every == 10_000


class TestPredictInterface:
    """Tests for the predict API (without training)."""

    def test_predict_requires_model(self, config):
        """predict() should raise error if model not trained."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")

        obs = np.zeros(560, dtype=np.float32)
        with pytest.raises(Exception):
            agent.predict(obs)

    def test_num_timesteps_zero_before_train(self, config):
        """num_timesteps should be 0 before training."""
        from agent.sac_agent import LatentAlphaSACAgent
        agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
        assert agent.num_timesteps == 0

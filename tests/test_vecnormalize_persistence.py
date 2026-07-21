"""Regression test: VecNormalize stats persist across save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import Env, spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class ConstantObsEnv(Env):
    """Minimal deterministic env with a fixed observation."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._obs = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._obs.copy(), {}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 10
        reward = float(np.tanh(action[0]))
        return self._obs.copy(), reward, terminated, False, {}


def _rollout_zero_actions(vec_env: VecNormalize, n_steps: int = 20) -> None:
    """Advance env so VecNormalize updates running stats."""
    obs = vec_env.reset()
    for _ in range(n_steps):
        zeros = np.zeros((vec_env.num_envs, vec_env.action_space.shape[0]), dtype=np.float32)
        obs, _, dones, _ = vec_env.step(zeros)
        if bool(dones[0]):
            obs = vec_env.reset()


@pytest.fixture
def config(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["model_dir"] = str(tmp_path / "models")
    cfg["sac"]["learning_rate"] = 1e-3
    cfg["sac"]["buffer_size"] = 256
    cfg["sac"]["batch_size"] = 32
    cfg["sac"]["learning_starts"] = 1
    cfg["sac"]["train_freq"] = 1
    cfg["sac"]["gradient_steps"] = 1
    cfg["sac"]["ent_coef"] = 0.02
    return cfg


def test_vecnormalize_stats_persist_and_reload(config):
    """Loaded VecNormalize should normalize raw obs identically."""
    from agent.sac_agent import LatentAlphaSACAgent

    train_vec = VecNormalize(
        DummyVecEnv([lambda: ConstantObsEnv()]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    agent = LatentAlphaSACAgent(config, risk_profile="balanced", term="medium")
    agent.model = SAC(
        policy="MlpPolicy",
        env=train_vec,
        learning_rate=1e-3,
        buffer_size=256,
        batch_size=32,
        learning_starts=1,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        ent_coef=0.02,
        verbose=0,
        seed=7,
        device="cpu",
    )
    agent.model.learn(total_timesteps=32)
    agent.save()

    raw_eval_env = DummyVecEnv([lambda: ConstantObsEnv()])
    agent.load(raw_env=raw_eval_env)
    loaded_vec = agent.vec_normalize
    assert isinstance(loaded_vec, VecNormalize)

    fresh_vec = VecNormalize(
        DummyVecEnv([lambda: ConstantObsEnv()]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    _rollout_zero_actions(fresh_vec, n_steps=32)

    loaded_vec.training = False
    loaded_vec.norm_reward = False
    fresh_vec.training = False
    fresh_vec.norm_reward = False

    raw_obs = np.array([[5.0, -3.0, 2.0]], dtype=np.float32)
    loaded_norm_obs = loaded_vec.normalize_obs(raw_obs.copy())
    fresh_norm_obs = fresh_vec.normalize_obs(raw_obs.copy())

    np.testing.assert_allclose(loaded_norm_obs, fresh_norm_obs, atol=1e-4, rtol=1e-4)

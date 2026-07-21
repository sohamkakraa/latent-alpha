"""
agent/ppo_agent.py
──────────────────
PPO agent wrapper around Stable-Baselines3.

Responsibilities:
  - Build the SB3 PPO model with architecture from config
  - Training with checkpoint saving (for nightly resume)
  - Inference / predict API for backtesting
  - Per-risk-profile model management

The agent is deliberately thin — all environment logic lives in
env/trading_env.py and reward shaping in agent/reward.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)


# ── Custom callbacks ──────────────────────────────────────────────────────────

class ActivityCheckCallback(BaseCallback):
    """
    Stops training if a sentinel file is present — used by the nightly
    scheduler to gracefully halt training when the user's machine becomes
    active in the morning (see scripts/night_trainer.py).
    """

    STOP_FILE = Path(".training_stop")

    def __init__(self, check_every_n_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.check_every = check_every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            if self.STOP_FILE.exists():
                logger.info("Stop file detected — pausing training gracefully.")
                return False  # Returning False stops training
        return True


class TrainingProgressCallback(BaseCallback):
    """Logs training progress to stdout at a configurable interval."""

    def __init__(self, log_every_n_steps: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            logger.info(
                "Training step %d | Mean reward: %.4f",
                self.num_timesteps,
                np.mean(self.locals.get("rewards", [0])),
            )
        return True


# ── Agent ─────────────────────────────────────────────────────────────────────

class LatentAlphaAgent:
    """
    Wrapper around SB3 PPO for the latent.alpha trading agent.

    Usage
    -----
    agent = LatentAlphaAgent(config, risk_profile="balanced", term="medium")

    # Train
    agent.train(train_env, eval_env=eval_env)

    # Save / load
    agent.save()
    agent.load()

    # Inference
    action, _ = agent.predict(obs)
    """

    def __init__(
        self,
        config: dict,
        risk_profile: str = "balanced",
        term: str = "medium",
        device: str = "auto",
        seed: int = 42,
    ):
        self.config       = config
        self.risk_profile = risk_profile
        self.term         = term
        self.seed         = int(seed)
        self.ppo_cfg      = config["ppo"]
        self.model_dir    = Path(config["paths"]["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path   = self.model_dir / f"ppo_{risk_profile}_{term}"

        # Device selection:
        # SB3 recommends CPU for MlpPolicy on discrete GPUs because PCIe
        # transfer overhead outweighs compute gains for small networks.
        # Apple M1/M2/M3 use unified memory (no transfer cost), so MPS is
        # worth trying — but CPU is still the safe default for small batches.
        # Run `python train.py --benchmark` to compare on your machine.
        if device == "auto":
            self.device = "cpu"  # safest default; override with --device mps
        else:
            self.device = device
        logger.info("Device: %s", self.device)

        logger.info("Agent initialised | risk=%s | term=%s | device=%s",
                    risk_profile, term, self.device)

        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        total_timesteps: Optional[int] = None,
        resume: bool = True,
    ) -> None:
        """
        Train the PPO agent.

        Parameters
        ----------
        train_env       : vectorised training environment
        eval_env        : optional evaluation environment for EvalCallback
        total_timesteps : overrides config value if provided
        resume          : if True and a checkpoint exists, resume from it
        """
        timesteps = total_timesteps or self.ppo_cfg["total_timesteps"]

        if resume and self._checkpoint_exists():
            logger.info("Resuming from checkpoint: %s", self.model_path)
            # custom_objects overrides hyperparameters saved in the checkpoint
            # with the current config values — so changes to ent_coef etc.
            # take effect immediately on resume without a full restart.
            self.model = PPO.load(
                str(self.model_path),
                env=train_env,
                device=self.device,
                custom_objects={
                    "ent_coef":       self.ppo_cfg["ent_coef"],
                    "learning_rate":  self.ppo_cfg["learning_rate"],
                    "clip_range":     self.ppo_cfg["clip_range"],
                    "max_grad_norm":  self.ppo_cfg["max_grad_norm"],
                },
            )
            if isinstance(train_env, VecNormalize):
                self.vec_normalize = train_env
            logger.info(
                "Hyperparameters overridden from config: ent_coef=%.4f lr=%.2e",
                self.ppo_cfg["ent_coef"], self.ppo_cfg["learning_rate"],
            )

            # If log_std has exploded (std >> 10), reset it to 0 (→ std=1.0).
            # This preserves all learned weights (value fn, policy mean) while
            # rescuing the action distribution from entropy collapse.
            if hasattr(self.model.policy, "log_std"):
                import torch
                current_std = self.model.policy.log_std.exp().mean().item()
                if current_std > 10.0:
                    logger.warning(
                        "log_std explosion detected (std=%.1f) — resetting to 0.0 (std=1.0). "
                        "Policy mean and value network weights are preserved.",
                        current_std,
                    )
                    with torch.no_grad():
                        self.model.policy.log_std.fill_(0.0)
        else:
            logger.info("Starting fresh training run.")
            self.model = PPO(
                policy=self.ppo_cfg["policy"],
                env=train_env,
                learning_rate=self.ppo_cfg["learning_rate"],
                n_steps=self.ppo_cfg["n_steps"],
                batch_size=self.ppo_cfg["batch_size"],
                n_epochs=self.ppo_cfg["n_epochs"],
                gamma=self.ppo_cfg["gamma"],
                gae_lambda=self.ppo_cfg["gae_lambda"],
                clip_range=self.ppo_cfg["clip_range"],
                ent_coef=self.ppo_cfg["ent_coef"],
                vf_coef=self.ppo_cfg["vf_coef"],
                max_grad_norm=self.ppo_cfg["max_grad_norm"],
                policy_kwargs={"net_arch": self.ppo_cfg["net_arch"]},
                tensorboard_log=self._tensorboard_log_dir(),
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
            if isinstance(train_env, VecNormalize):
                self.vec_normalize = train_env

        callbacks = [
            # Save checkpoint every 10k steps — enables nightly resume
            CheckpointCallback(
                save_freq=10_000,
                save_path=str(self.model_dir / "checkpoints"),
                name_prefix=f"ppo_{self.risk_profile}_{self.term}",
            ),
            # Halt training if stop file is detected (morning activity)
            ActivityCheckCallback(check_every_n_steps=self.ppo_cfg["n_steps"]),
            TrainingProgressCallback(log_every_n_steps=10_000),
        ]

        if eval_env is not None:
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.model_dir / "best"),
                    log_path=str(Path(self.config["paths"]["logs_dir"]) / "eval"),
                    eval_freq=max(10_000 // 1, 1),   # evaluate every 10k steps
                    n_eval_episodes=3,
                    deterministic=True,
                )
            )

        # When resuming, only train for the *remaining* steps so the progress
        # bar shows 0 → remaining (e.g. 0 → 3.7M) instead of always 0 → 5M.
        already_done = self.model.num_timesteps if (resume and self._checkpoint_exists()) else 0
        remaining_timesteps = max(timesteps - already_done, 0)

        self.model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=not resume,
            progress_bar=True,
        )
        self.save()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Predict action for a given observation.

        Parameters
        ----------
        obs           : observation array from the environment
        deterministic : if True, use the mean of the policy (no exploration)

        Returns
        -------
        (action, state) — state is always None for PPO (no RNN)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        return self.model.predict(obs, deterministic=deterministic)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(str(self.model_path))
        env = self.model.get_env()
        vecnorm = env if isinstance(env, VecNormalize) else self.vec_normalize
        if isinstance(vecnorm, VecNormalize):
            vecnorm.save(str(self._vecnorm_stats_path()))
        logger.info("Model saved to %s", self.model_path)

    def load(self, raw_env: Optional[VecEnv] = None) -> None:
        if not self._checkpoint_exists():
            raise FileNotFoundError(f"No model found at {self.model_path}")
        self.model = PPO.load(str(self.model_path), device=self.device)
        stats_path = self._vecnorm_stats_path()
        if raw_env is not None and stats_path.exists():
            self.vec_normalize = VecNormalize.load(str(stats_path), raw_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            self.model.set_env(self.vec_normalize)
        elif raw_env is not None:
            self.model.set_env(raw_env)
        logger.info("Model loaded from %s", self.model_path)

    def _checkpoint_exists(self) -> bool:
        return Path(str(self.model_path) + ".zip").exists()

    def _vecnorm_stats_path(self) -> Path:
        return self.model_path.with_suffix(".vecnorm.pkl")

    def _tensorboard_log_dir(self):
        """Return tensorboard log dir only if tensorboard is installed, else None."""
        try:
            import tensorboard  # noqa: F401
            log_dir = Path(self.config["paths"]["logs_dir"]) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            return str(log_dir)
        except ImportError:
            return None

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps if self.model else 0

    def __repr__(self) -> str:
        return (
            f"LatentAlphaAgent(risk={self.risk_profile}, term={self.term}, "
            f"device={self.device}, trained_steps={self.num_timesteps})"
        )

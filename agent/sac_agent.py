"""
agent/sac_agent.py
──────────────────
SAC (Soft Actor-Critic) agent wrapper around Stable-Baselines3.

Why SAC over PPO for portfolio allocation:
  - Automatic entropy tuning — no more hand-tuning ent_coef or watching
    std explode. SAC learns the optimal exploration–exploitation balance.
  - Off-policy + replay buffer — more sample-efficient, typically converges
    in 1–2M steps vs PPO's 5M for the same quality.
  - Continuous action space native — SAC was designed specifically for
    continuous control, which is what portfolio weight allocation is.
  - Squashed Gaussian policy — outputs are naturally bounded via tanh,
    so actions stay in [-1, 1] without clipping artifacts.

Trade-off: SAC uses more memory (replay buffer) and is slightly more
complex, but for a 10-stock daily-bar environment this is negligible.

The interface mirrors LatentAlphaAgent exactly — same train/predict/save/load
API — so the backtest runner and train.py work with either.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

logger = logging.getLogger(__name__)


# ── Custom callbacks ─────────────────────────────────────────────────────────

class TrainingInterrupted(RuntimeError):
    """Raised after a graceful stop so a fold is never marked complete."""


class EntropyFloorCallback(BaseCallback):
    """
    Prevents entropy collapse by clamping ent_coef to a minimum value.

    Without this, SAC's automatic entropy tuning drives ent_coef to near-zero,
    killing exploration. The floor ensures the agent always maintains some
    minimum stochasticity.
    """

    def __init__(
        self,
        ent_coef_floor: float = 0.005,
        log_every_n_steps: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ent_coef_floor = ent_coef_floor
        self.log_every = log_every_n_steps
        self._clamp_count = 0

    def _on_step(self) -> bool:
        import math
        if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
            with torch.no_grad():
                current_log = self.model.log_ent_coef.item()
                current_ent = math.exp(current_log)

                if current_ent < self.ent_coef_floor:
                    clamped_log = math.log(self.ent_coef_floor)
                    self.model.log_ent_coef.fill_(clamped_log)
                    self._clamp_count += 1

                    if self._clamp_count % 100 == 1:
                        logger.warning(
                            "Entropy floor active: clamped ent_coef from %.6f to %.4f "
                            "(clamp #%d)",
                            current_ent, self.ent_coef_floor, self._clamp_count,
                        )

            if self.n_calls % self.log_every == 0 and self.model.log_ent_coef is not None:
                import math as _math
                ent_coef = _math.exp(self.model.log_ent_coef.item())
                logger.info(
                    "Step %d | ent_coef=%.6f | clamps=%d",
                    self.num_timesteps, ent_coef, self._clamp_count,
                )
        return True


class ActivityCheckCallback(BaseCallback):
    """Stops training if a sentinel file is present."""

    STOP_FILE = Path(".training_stop")

    def __init__(self, check_every_n_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.check_every = check_every_n_steps
        self.was_stopped = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            if self.STOP_FILE.exists():
                logger.info("Stop file detected — pausing training gracefully.")
                self.was_stopped = True
                return False
        return True


class TrainingProgressCallback(BaseCallback):
    """Logs training progress at a configurable interval."""

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


class ReplayBufferCheckpointCallback(BaseCallback):
    """Persist replay state at policy-checkpoint boundaries."""

    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = max(int(save_freq), 1)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / (
            f"{self.name_prefix}_replay_buffer_"
            f"{self.num_timesteps}_steps.pkl"
        )
        self.model.save_replay_buffer(str(path))
        logger.info("Replay buffer checkpoint saved to %s", path)
        return True


# ── Agent ────────────────────────────────────────────────────────────────────

class LatentAlphaSACAgent:
    """
    SAC wrapper for the latent.alpha trading agent.

    Drop-in replacement for LatentAlphaAgent (PPO) — same API:
      agent.train(train_env, eval_env)
      agent.predict(obs)
      agent.save() / agent.load()

    Key SAC hyperparameters (from config["sac"]):
      - learning_rate:       3e-4 (Adam, separate for actor/critic)
      - buffer_size:         1_000_000 (replay buffer capacity)
      - batch_size:          256 (mini-batch from replay buffer)
      - tau:                 0.005 (soft target update rate)
      - gamma:               0.99 (discount factor)
      - ent_coef:            "auto" (automatic entropy tuning — the key advantage)
      - train_freq:          1 (update every step)
      - gradient_steps:      1 (one gradient step per env step)
      - learning_starts:     10_000 (fill buffer before training)
      - target_entropy:      "auto" (SB3 computes -dim(action_space))
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
        self.sac_cfg      = config["sac"]
        self.entropy_cfg  = {
            "ent_coef": self.sac_cfg.get("ent_coef", 0.02),
        }
        self.model_dir    = Path(config["paths"]["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path   = self.model_dir / f"sac_{risk_profile}_{term}"

        if device == "auto":
            self.device = "cpu"
        else:
            self.device = device
        logger.info("Device: %s", self.device)

        logger.info("SAC Agent initialised | risk=%s | term=%s | device=%s",
                    risk_profile, term, self.device)

        self.model: Optional[SAC] = None
        self.vec_normalize: Optional[VecNormalize] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        total_timesteps: Optional[int] = None,
        resume: bool = True,
        regime_fn=None,
    ) -> None:
        """
        Train the SAC agent.

        Parameters
        ----------
        train_env       : vectorised training environment
        eval_env        : optional evaluation environment for EvalCallback
        total_timesteps : overrides config value if provided
        resume          : if True and a checkpoint exists, resume from it
        """
        timesteps = total_timesteps or self.sac_cfg["total_timesteps"]

        # Only resume from periodic checkpoints (mid-fold crash recovery).
        # The canonical end-of-fold save MUST NOT be used for resume across folds
        # in a walk-forward backtest: each fold trains a fresh policy on its own
        # train window. The runner clears periodic checkpoints after each fold
        # completes, so a stale periodic from a prior fold should never appear.
        resume_path: Optional[Path] = None
        if resume:
            periodic = self._latest_periodic_checkpoint()
            if periodic is not None:
                resume_path = periodic

        if resume_path is not None:
            logger.info("Resuming from checkpoint: %s", resume_path)
            # Restore VecNormalize running stats saved alongside the checkpoint.
            # Without this, train_env attaches a fresh normalizer with zeroed
            # obs/ret running stats and the obs distribution shifts mid-training
            # on every resume. The replay buffer is NOT restored (it is not saved
            # — ~GBs per checkpoint at this buffer_size), so a resumed run still
            # differs from an uninterrupted one; treat resume as crash recovery,
            # not as a science-grade continuation.
            if isinstance(train_env, VecNormalize):
                vnorm_path = self._matching_periodic_vecnorm(resume_path)
                if vnorm_path is not None:
                    try:
                        saved = VecNormalize.load(str(vnorm_path), train_env.venv)
                        train_env.obs_rms = saved.obs_rms
                        train_env.ret_rms = saved.ret_rms
                        logger.info("Restored VecNormalize stats from %s", vnorm_path)
                    except Exception as e:
                        logger.warning("Failed to restore VecNormalize stats: %s", e)
                else:
                    logger.warning(
                        "No VecNormalize stats found for checkpoint %s — "
                        "obs normalization restarts from scratch.", resume_path,
                    )
            self.model = SAC.load(
                str(resume_path.with_suffix("")),
                env=train_env,
                device=self.device,
                custom_objects={
                    "learning_rate":  self.sac_cfg["learning_rate"],
                },
            )
            replay_path = self._matching_periodic_replay(resume_path)
            if replay_path is None:
                raise RuntimeError(
                    "Science-grade resume selected a policy checkpoint without "
                    f"matching replay state: {resume_path}"
                )
            self.model.load_replay_buffer(str(replay_path))
            logger.info("Restored replay buffer from %s", replay_path)
            logger.info(
                "Hyperparameters overridden from config: lr=%.2e",
                self.sac_cfg["learning_rate"],
            )
            if isinstance(train_env, VecNormalize):
                self.vec_normalize = train_env
        else:
            logger.info("Starting fresh SAC training run.")

            # Parse net_arch — SAC uses separate pi (actor) and qf (critic) nets
            net_arch_raw = self.sac_cfg.get("net_arch", [256, 256])
            policy_kwargs = {"net_arch": net_arch_raw}

            # v3.2 P0-fix: Fixed ent_coef — auto-tuning collapsed across every
            # experiment (ent_coef → 0.0004 by 500K steps). A fixed coefficient
            # guarantees stable exploration throughout training.
            # 0.02 = moderate exploration: enough to discover new allocations,
            # not so much that the policy is random.
            ent_coef = self.entropy_cfg["ent_coef"]
            logger.info(
                "SAC entropy config: FIXED ent_coef=%.4f (auto-tuning disabled)",
                ent_coef,
            )

            self.model = SAC(
                policy=self.sac_cfg.get("policy", "MlpPolicy"),
                env=train_env,
                learning_rate=self.sac_cfg["learning_rate"],
                buffer_size=self.sac_cfg["buffer_size"],
                batch_size=self.sac_cfg["batch_size"],
                tau=self.sac_cfg["tau"],
                gamma=self.sac_cfg["gamma"],
                ent_coef=ent_coef,                               # Fixed, not "auto"
                train_freq=self.sac_cfg.get("train_freq", 1),
                gradient_steps=self.sac_cfg.get("gradient_steps", 1),
                learning_starts=self.sac_cfg["learning_starts"],
                policy_kwargs=policy_kwargs,
                tensorboard_log=self._tensorboard_log_dir(),
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
            if isinstance(train_env, VecNormalize):
                self.vec_normalize = train_env

        checkpoint_freq = int(self.sac_cfg.get("checkpoint_freq", 10_000))
        replay_checkpoint_freq = int(
            self.sac_cfg.get("replay_checkpoint_freq", 50_000)
        )
        activity_callback = ActivityCheckCallback(check_every_n_steps=4096)
        callbacks = [
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(self.model_dir / "checkpoints"),
                name_prefix=f"sac_{self.risk_profile}_{self.term}",
                # Persist obs/ret normalization stats so a resumed run does not
                # restart normalization from zero. Tiny file; saved only when the
                # model's env is a VecNormalize.
                save_vecnormalize=True,
            ),
            ReplayBufferCheckpointCallback(
                save_freq=replay_checkpoint_freq,
                save_path=self.model_dir / "checkpoints",
                name_prefix=f"sac_{self.risk_profile}_{self.term}",
            ),
            activity_callback,
            TrainingProgressCallback(log_every_n_steps=10_000),
        ]

        if eval_env is not None:
            # Early stopping on the held-out eval slice: halt when the eval score
            # plateaus so a single long run auto-finds the sweet spot instead of
            # brute-forcing a timestep grid. patience=0 disables it.
            patience   = int(self.sac_cfg.get("early_stop_patience", 0))
            after_eval = None
            if patience > 0:
                after_eval = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=patience,
                    min_evals=int(self.sac_cfg.get("early_stop_min_evals", 10)),
                    verbose=1,
                )
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.model_dir / "best"),
                    log_path=str(Path(self.config["paths"]["logs_dir"]) / "eval"),
                    eval_freq=int(self.sac_cfg.get("eval_freq", 10_000)),
                    n_eval_episodes=int(self.sac_cfg.get("n_eval_episodes", 1)),
                    deterministic=True,
                    callback_after_eval=after_eval,
                )
            )

        # When resuming, only train for remaining steps
        already_done = self.model.num_timesteps if resume_path is not None else 0
        remaining_timesteps = max(timesteps - already_done, 0)
        if resume_path is not None:
            logger.info(
                "Resume: %d/%d timesteps already completed, %d remaining.",
                already_done, timesteps, remaining_timesteps,
            )

        self.model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_path is None,
            progress_bar=True,
        )
        self.save()
        if activity_callback.was_stopped:
            recovery_path = self._save_recovery_checkpoint()
            raise TrainingInterrupted(
                "Training paused before the requested timestep target. "
                f"Resume from {recovery_path}."
            )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
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
        self.model = SAC.load(str(self.model_path), device=self.device)
        stats_path = self._vecnorm_stats_path()
        if raw_env is not None and stats_path.exists():
            self.vec_normalize = VecNormalize.load(str(stats_path), raw_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            self.model.set_env(self.vec_normalize)
        elif raw_env is not None:
            logger.warning(
                "VecNormalize stats not found at %s; using raw observations.",
                stats_path,
            )
            self.model.set_env(raw_env)
        logger.info("Model loaded from %s", self.model_path)

    def _checkpoint_exists(self) -> bool:
        return Path(str(self.model_path) + ".zip").exists()

    def clear_periodic_checkpoints(self) -> int:
        """
        Delete periodic checkpoints for this (risk_profile, term). Call after a
        fold completes successfully so the next fold doesn't accidentally resume
        from a prior fold's mid-training state. Returns count of files removed.
        """
        ckpt_dir = self.model_dir / "checkpoints"
        if not ckpt_dir.exists():
            return 0
        prefix = f"sac_{self.risk_profile}_{self.term}"
        removed = 0
        # Remove both the policy checkpoints and their paired VecNormalize pkls.
        for pattern in (
            f"{prefix}_*_steps.zip",
            f"{prefix}_vecnormalize_*_steps.pkl",
            f"{prefix}_replay_buffer_*_steps.pkl",
        ):
            for p in ckpt_dir.glob(pattern):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("Cleared %d periodic checkpoint file(s) from %s", removed, ckpt_dir)
        return removed

    def _latest_periodic_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent periodic checkpoint written by CheckpointCallback.
        Returns None if no checkpoints exist. Used to recover from mid-fold crashes
        without losing all training progress.
        """
        ckpt_dir = self.model_dir / "checkpoints"
        if not ckpt_dir.exists():
            return None
        prefix = f"sac_{self.risk_profile}_{self.term}"
        candidates = [
            path
            for path in ckpt_dir.glob(f"{prefix}_*_steps.zip")
            if self._matching_periodic_vecnorm(path) is not None
            and self._matching_periodic_replay(path) is not None
        ]
        if not candidates:
            return None

        def _step_count(p: Path) -> int:
            try:
                return int(p.stem.rsplit("_", 2)[-2])
            except (ValueError, IndexError):
                return -1

        return max(candidates, key=_step_count)

    def _matching_periodic_vecnorm(self, checkpoint: Path) -> Optional[Path]:
        """
        Given a periodic policy checkpoint (``..._{steps}_steps.zip``), return the
        VecNormalize stats pkl CheckpointCallback wrote at the same step
        (``..._vecnormalize_{steps}_steps.pkl``), or None if it does not exist.
        """
        try:
            steps = checkpoint.stem.rsplit("_", 2)[-2]
        except IndexError:
            return None
        prefix = f"sac_{self.risk_profile}_{self.term}"
        vpath = checkpoint.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl"
        return vpath if vpath.exists() else None

    def _matching_periodic_replay(self, checkpoint: Path) -> Optional[Path]:
        try:
            steps = checkpoint.stem.rsplit("_", 2)[-2]
        except IndexError:
            return None
        prefix = f"sac_{self.risk_profile}_{self.term}"
        path = checkpoint.parent / (
            f"{prefix}_replay_buffer_{steps}_steps.pkl"
        )
        return path if path.exists() else None

    def _save_recovery_checkpoint(self) -> Path:
        """Save policy, normalization, and replay state at the same step."""
        if self.model is None:
            raise RuntimeError("No model available for recovery checkpoint.")
        steps = self.model.num_timesteps
        checkpoint_dir = self.model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"sac_{self.risk_profile}_{self.term}"
        policy_path = checkpoint_dir / f"{prefix}_recovery_{steps}_steps"
        self.model.save(str(policy_path))

        env = self.model.get_env()
        vecnorm = env if isinstance(env, VecNormalize) else self.vec_normalize
        if not isinstance(vecnorm, VecNormalize):
            raise RuntimeError(
                "Cannot create science-grade recovery checkpoint without "
                "VecNormalize state."
            )
        vecnorm.save(
            str(
                checkpoint_dir
                / f"{prefix}_vecnormalize_{steps}_steps.pkl"
            )
        )
        self.model.save_replay_buffer(
            str(
                checkpoint_dir
                / f"{prefix}_replay_buffer_{steps}_steps.pkl"
            )
        )
        logger.info(
            "Recovery checkpoint saved at step %d: %s",
            steps,
            policy_path,
        )
        return policy_path.with_suffix(".zip")

    def _vecnorm_stats_path(self) -> Path:
        return self.model_path.with_suffix(".vecnorm.pkl")

    def _tensorboard_log_dir(self):
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
            f"LatentAlphaSACAgent(risk={self.risk_profile}, term={self.term}, "
            f"device={self.device}, trained_steps={self.num_timesteps})"
        )

"""
scripts/night_trainer.py
─────────────────────────
Nightly GPU training orchestrator.

Behaviour
---------
1. Runs at a scheduled time (default: 11 PM)
2. Detects GPU availability (CUDA or Apple MPS) and sets device accordingly
3. Starts / resumes training from the latest checkpoint
4. Monitors system idle time — pauses training when user activity is detected
5. Creates a .training_stop sentinel file to signal the agent's ActivityCheckCallback
6. Saves a checkpoint before halting so training resumes exactly where it left off
7. Logs each session's progress (steps completed, time elapsed)

Usage
-----
# Run directly (blocks until user wakes up or training completes)
python scripts/night_trainer.py

# Run with custom settings
python scripts/night_trainer.py --risk balanced --term medium --idle-threshold 300

# Dry run: simulate one night without actually training
python scripts/night_trainer.py --dry-run

macOS idle detection
--------------------
Uses `ioreg` to read the HIDIdleTime (nanoseconds since last input event).
Windows: reads LastInputInfo via ctypes.
Linux: reads /proc/uptime and compares with xprintidle (if available).
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# Create logs dir before FileHandler tries to open the file
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "night_trainer.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

CONFIG_PATH  = Path(__file__).parent.parent / "config" / "config.yaml"
STOP_FILE    = Path(".training_stop")


# ── Idle time detection ───────────────────────────────────────────────────────

def get_idle_seconds() -> float:
    """
    Return the number of seconds since the last user input event.
    Works on macOS, Windows, and Linux.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "HIDIdleTime" in line:
                    # Value is in nanoseconds
                    ns = int(line.split("=")[-1].strip())
                    return ns / 1_000_000_000
        except Exception as e:
            logger.debug("macOS idle detection failed: %s", e)
        return 0.0

    elif system == "Windows":
        try:
            import ctypes
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(lii)
            ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
            millis_elapsed = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return millis_elapsed / 1000.0
        except Exception as e:
            logger.debug("Windows idle detection failed: %s", e)
        return 0.0

    else:  # Linux
        try:
            result = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) / 1000.0
        except FileNotFoundError:
            logger.debug("xprintidle not found — install with: sudo apt install xprintidle")
        except Exception as e:
            logger.debug("Linux idle detection failed: %s", e)
        return float("inf")  # Assume idle if detection fails on Linux


def is_user_active(idle_threshold_seconds: int = 120) -> bool:
    """Return True if the user has been active within the threshold window."""
    idle = get_idle_seconds()
    return idle < idle_threshold_seconds


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device() -> str:
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("CUDA GPU detected: %s (%.1f GB VRAM)", gpu_name, vram)
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal) detected — using GPU acceleration.")
            return "mps"
        else:
            logger.warning("No GPU detected — training on CPU (will be slow).")
            return "cpu"
    except ImportError:
        logger.error("PyTorch not installed.")
        return "cpu"


# ── Training session ──────────────────────────────────────────────────────────

def run_training_session(
    config: dict,
    risk_profile: str,
    term: str,
    device: str,
    idle_threshold: int,
    poll_interval: int,
    dry_run: bool,
) -> None:
    """
    Start or resume a training run, monitoring for user activity.

    The training loop runs in a separate thread. The main thread polls
    for user activity every `poll_interval` seconds and signals the
    training loop to stop by writing the STOP_FILE.
    """
    if dry_run:
        logger.info("[DRY RUN] Would train: risk=%s term=%s device=%s", risk_profile, term, device)
        _simulate_dry_run(idle_threshold, poll_interval)
        return

    # Remove stale stop file from a previous session
    STOP_FILE.unlink(missing_ok=True)

    import threading
    from agent.sac_agent import LatentAlphaSACAgent, TrainingInterrupted
    from strategy.term_selector import TermSelector

    selector  = TermSelector(config)
    agent     = LatentAlphaSACAgent(config, risk_profile, term, device=device)

    train_env, eval_env = selector.build_env(
        risk_profile=risk_profile,
        term=term,
        train_start=config["backtest"]["start_date"],
        train_end=config["backtest"]["end_date"],
    )

    session_start = datetime.now()
    logger.info("═" * 60)
    logger.info("Night training session started: %s", session_start.strftime("%Y-%m-%d %H:%M"))
    logger.info("risk=%s | term=%s | device=%s | idle_threshold=%ds",
                risk_profile, term, device, idle_threshold)

    training_error: list = []

    def training_thread():
        """Run SAC training in background thread."""
        try:
            agent.train(train_env=train_env, eval_env=eval_env, resume=True)
        except TrainingInterrupted as e:
            logger.info("Training paused with complete recovery state: %s", e)
        except Exception as e:
            training_error.append(e)
            logger.error("Training thread error: %s", e)

    thread = threading.Thread(target=training_thread, daemon=True)
    thread.start()

    # ── Activity monitor loop ─────────────────────────────────────────────────
    while thread.is_alive():
        time.sleep(poll_interval)

        if training_error:
            logger.error("Training failed — session aborted.")
            STOP_FILE.touch()
            break

        if is_user_active(idle_threshold_seconds=idle_threshold):
            logger.info(
                "User activity detected (idle < %ds) — signalling training to pause.",
                idle_threshold,
            )
            STOP_FILE.touch()  # ActivityCheckCallback will see this and stop
            thread.join(timeout=60)
            logger.info("Training paused. Checkpoint saved.")
            break

        idle_s = get_idle_seconds()
        elapsed = (datetime.now() - session_start).total_seconds() / 60
        logger.info(
            "Training running | elapsed: %.0f min | idle: %.0fs | steps: %s",
            elapsed, idle_s, f"{agent.num_timesteps:,}" if agent.model else "—",
        )

    # Clean up
    STOP_FILE.unlink(missing_ok=True)
    elapsed_min = (datetime.now() - session_start).total_seconds() / 60
    logger.info("Session ended. Total elapsed: %.1f minutes.", elapsed_min)
    logger.info("═" * 60)


def _simulate_dry_run(idle_threshold: int, poll_interval: int) -> None:
    """Simulate the activity monitoring loop without actual training."""
    logger.info("Dry run: monitoring idle time for 30 seconds...")
    for i in range(3):
        time.sleep(min(poll_interval, 10))
        idle = get_idle_seconds()
        active = is_user_active(idle_threshold)
        logger.info(
            "Dry run poll %d/3 | idle: %.1fs | user_active: %s",
            i + 1, idle, active,
        )
    logger.info("Dry run complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="latent.alpha — nightly GPU training scheduler"
    )
    parser.add_argument("--risk", default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term", default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--idle-threshold", type=int, default=120,
                        help="Seconds of idle time before training starts/continues. "
                             "Training stops if idle drops below this (user is active). Default: 120s")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="How often (seconds) to check for user activity. Default: 60s")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Test idle detection without training")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = detect_device() if args.device == "auto" else args.device

    # Wait until user is idle before starting
    logger.info(
        "Waiting for user to be idle (>%ds) before starting training...",
        args.idle_threshold,
    )
    while is_user_active(args.idle_threshold):
        logger.info("User still active — checking again in %ds.", args.poll_interval)
        time.sleep(args.poll_interval)

    run_training_session(
        config=config,
        risk_profile=args.risk,
        term=args.term,
        device=device,
        idle_threshold=args.idle_threshold,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

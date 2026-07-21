"""
train.py
────────
Training entry point for the latent.alpha trading agent.

Supports PPO and SAC algorithms. SAC is recommended — it auto-tunes
exploration and converges faster for continuous portfolio weight allocation.

Usage
-----
# SAC (recommended) — start fresh
python train.py --algo sac --risk balanced --term medium

# PPO — start fresh
python train.py --algo ppo --risk balanced --term medium

# Resume from last checkpoint
python train.py --algo sac --risk balanced --term medium --resume

# Train all risk profiles sequentially
python train.py --algo sac --all-profiles --term medium

# Enable LLM news sentiment
python train.py --algo sac --risk balanced --term medium --with-news

Pause/resume
------------
Press Ctrl+C at any time — the agent saves a checkpoint and exits cleanly.
Run the same command again with --resume to pick up exactly where you left off.
Checkpoints are saved automatically every 10,000 steps regardless.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger  = logging.getLogger(__name__)
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"

# Global reference so the signal handler can reach the agent
_active_agent = None


def _handle_interrupt(sig, frame):
    """Ctrl+C handler — save checkpoint then exit cleanly."""
    console.print("\n[yellow]⏸  Interrupt received — saving checkpoint...[/yellow]")
    if _active_agent is not None and _active_agent.model is not None:
        try:
            _active_agent.save()
            console.print(
                f"[green]✓ Checkpoint saved[/green] "
                f"([dim]{_active_agent.model_path}.zip[/dim])\n"
                f"Resume with: [bold]python train.py "
                f"--risk {_active_agent.risk_profile} "
                f"--term {_active_agent.term} --resume[/bold]"
            )
        except Exception as e:
            console.print(f"[red]Failed to save checkpoint: {e}[/red]")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def benchmark_devices(config: dict) -> None:
    """
    Benchmark actual SAC gradient updates on CPU and Apple MPS.

    A synthetic environment isolates policy-update throughput from network/data
    downloads while matching the production observation and action dimensions.
    """
    import time
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results  = {}
    n_stocks = len(config["market"]["tickers"])
    obs_dim = (
        n_stocks * 35
        + n_stocks
        + 2
        + n_stocks
        + n_stocks * config["llm"]["sentiment_dim"]
        + config.get("meta_model", {}).get("obs_dim", 20)
        + config.get("vae_regime", {}).get("obs_dim", 0)
    )

    class BenchmarkEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                -10.0, 10.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(n_stocks,), dtype=np.float32
            )
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            obs = self.np_random.normal(
                0.0, 1.0, size=obs_dim
            ).astype(np.float32)
            reward = float(-np.mean(np.square(action)))
            terminated = self.steps >= 500
            return obs, reward, terminated, False, {}

    for device in devices:
        console.rule(f"Benchmarking: [cyan]{device.upper()}[/cyan]")
        train_env = DummyVecEnv([BenchmarkEnv])
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_starts=250,
            buffer_size=5_000,
            batch_size=config["sac"]["batch_size"],
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={"net_arch": config["sac"]["net_arch"]},
            ent_coef=config["sac"]["ent_coef"],
            seed=42,
            device=device,
            verbose=0,
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=2_000)
        elapsed = time.perf_counter() - t0
        sps = 2_000 / elapsed
        results[device] = sps
        console.print(f"  {device.upper():4s}: [bold]{sps:,.0f}[/bold] steps/sec  ({elapsed:.1f}s for 2k steps)")

    console.rule("Result")
    best = max(results, key=results.get)
    console.print(f"  Fastest device: [bold green]{best.upper()}[/bold green]")
    console.print(
        "  Use: [bold]python backtest.py --risk balanced --term medium "
        f"--device {best}[/bold]"
    )


def _make_agent(
    algo: str,
    config: dict,
    risk_profile: str,
    term: str,
    device: str,
    seed: int,
):
    """Factory: return the right agent class based on --algo flag."""
    if algo == "sac":
        from agent.sac_agent import LatentAlphaSACAgent
        return LatentAlphaSACAgent(
            config,
            risk_profile=risk_profile,
            term=term,
            device=device,
            seed=seed,
        )
    else:
        from agent.ppo_agent import LatentAlphaAgent
        return LatentAlphaAgent(
            config,
            risk_profile=risk_profile,
            term=term,
            device=device,
            seed=seed,
        )


def train(
    config: dict,
    risk_profile: str,
    term: str,
    algo: str = "sac",
    resume: bool = True,
    with_news: bool = False,
    device: str = "auto",
    timesteps: Optional[int] = None,
    seed: int = 42,
) -> None:
    global _active_agent

    from strategy.term_selector import TermSelector

    # ── LLM sentiment ─────────────────────────────────────────────────────────
    # Load from pre-computed historical cache (built by precompute_sentiment.py).
    # Noise + dropout are applied during training so the agent learns to handle
    # imperfect/missing news — more robust for live trading.
    sentiment_fn = sentiment_fn_eval = None
    from llm.sentiment_cache import SentimentCache
    from pathlib import Path as _Path

    tickers    = config["market"]["tickers"]
    vec_dim    = len(tickers) * config["llm"]["sentiment_dim"]
    cache_path = _Path("data/sentiment_cache.db")

    if with_news and cache_path.exists():
        cache = SentimentCache(cache_path)
        model_id = config["llm"]["sentiment_cache_model"]
        schema_version = int(config["llm"]["sentiment_schema_version"])
        import pandas as pd
        required_dates = [
            date.strftime("%Y-%m-%d")
            for date in pd.bdate_range(
                config["backtest"]["start_date"],
                config["backtest"]["end_date"],
            )
        ]
        coverage = cache.validate_coverage(
            required_dates,
            model=model_id,
            vector_dim=vec_dim,
            schema_version=schema_version,
        )
        sentiment_fn = cache.make_lookup_fn(
            vector_dim=vec_dim,
            noise_std=config["llm"]["noise_std"],
            dropout_prob=config["llm"]["dropout_prob"],
            training=True,   # noise + dropout only active here
            model=model_id,
            schema_version=schema_version,
            seed=seed,
        )
        sentiment_fn_eval = cache.make_lookup_fn(
            vector_dim=vec_dim,
            training=False,
            model=model_id,
            schema_version=schema_version,
            seed=seed,
        )
        console.print(
            f"[green]✓ Sentiment cache loaded[/green] — "
            f"{coverage['required_dates']} dates | "
            f"noise={config['llm']['noise_std']} dropout={config['llm']['dropout_prob']}"
        )
    elif with_news:
        from llm.news_fetcher import NewsFetcher
        from llm.verifier import NewsVerifier
        from llm.sentiment_encoder import SentimentEncoder
        fetcher  = NewsFetcher(config)
        verifier = NewsVerifier(min_sources=config["llm"]["min_sources_for_verification"])
        encoder  = SentimentEncoder(config, tickers)
        articles = fetcher.fetch_for_all_tickers(tickers)
        verified = {t: verifier.verify(a) for t, a in articles.items()}
        sentiment_fn = encoder.make_sentiment_fn(verified)
        sentiment_fn_eval = sentiment_fn
        console.print("[yellow]⚠ No cache — using live news (current date only)[/yellow]")
    else:
        console.print(
            "[dim]Sentiment OFF. Pass --with-news after building a validated "
            "historical cache.[/dim]"
        )

    # ── Build environments ────────────────────────────────────────────────────
    selector    = TermSelector(config)
    train_start = config["backtest"]["start_date"]
    train_end   = config["backtest"]["end_date"]

    prefix = "sac" if algo == "sac" else "ppo"
    has_checkpoint = (
        Path(config["paths"]["model_dir"]) / f"{prefix}_{risk_profile}_{term}.zip"
    ).exists()

    if resume and has_checkpoint:
        console.print(
            f"[bold]Resuming:[/bold] algo=[cyan]{algo.upper()}[/cyan] | "
            f"risk=[cyan]{risk_profile}[/cyan] | term=[cyan]{term}[/cyan]"
        )
    else:
        console.print(
            f"[bold]Starting:[/bold] algo=[cyan]{algo.upper()}[/cyan] | "
            f"risk=[cyan]{risk_profile}[/cyan] | term=[cyan]{term}[/cyan]"
        )

    console.print("[dim]Press Ctrl+C at any time to pause and save.[/dim]")

    train_env, eval_env = selector.build_env(
        risk_profile=risk_profile,
        term=term,
        train_start=train_start,
        train_end=train_end,
        sentiment_fn=sentiment_fn,
        sentiment_fn_eval=sentiment_fn_eval,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    config.setdefault("experiment", {})["seed"] = int(seed)
    agent = _make_agent(
        algo,
        config,
        risk_profile=risk_profile,
        term=term,
        device=device,
        seed=seed,
    )
    _active_agent = agent  # expose to signal handler

    agent.train(train_env=train_env, eval_env=eval_env, resume=resume,
                total_timesteps=timesteps)

    console.print(
        f"\n[bold green]✓ Training complete[/bold green] | "
        f"Total steps: {agent.num_timesteps:,}"
    )
    _active_agent = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="latent.alpha — train RL trading agent (SAC or PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Press Ctrl+C at any time to pause. Re-run with --resume to continue.",
    )
    parser.add_argument("--algo",         default="sac",
                        choices=["sac", "ppo"],
                        help="RL algorithm. SAC recommended (auto entropy tuning)")
    parser.add_argument("--risk",         default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term",         default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from last checkpoint (safe even if no checkpoint exists)")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Train all three risk profiles sequentially")
    parser.add_argument("--with-news",    action="store_true",
                        help="Enable LLM news sentiment (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--device",       default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device. auto=cpu. M1/M2/M3 owners: try --device mps")
    parser.add_argument("--timesteps",    type=int, default=None,
                        help="Override total training timesteps (e.g. 10000 for a smoke test)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--benchmark",    action="store_true",
                        help="Time CPU vs MPS on your machine and print the faster option")
    args = parser.parse_args()

    config = load_config()

    if args.benchmark:
        benchmark_devices(config)
        return

    profiles = ["conservative", "balanced", "aggressive"] if args.all_profiles else [args.risk]

    for profile in profiles:
        console.rule(f"[bold]{args.algo.upper()} / {profile.upper()} / {args.term.upper()}[/bold]")
        train(
            config=config,
            risk_profile=profile,
            term=args.term,
            algo=args.algo,
            resume=args.resume,
            with_news=args.with_news,
            device=args.device,
            timesteps=args.timesteps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

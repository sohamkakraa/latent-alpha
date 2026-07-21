"""
backtest.py
───────────
Walk-forward backtesting entry point.

Usage
-----
# Backtest balanced agent on medium term
python backtest.py

# Backtest all profiles
python backtest.py --all-profiles --term medium

# Backtest without retraining (uses existing model checkpoints)
python backtest.py --risk aggressive --term long --no-retrain
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger  = logging.getLogger(__name__)
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_backtest(
    config: dict,
    risk_profile: str,
    term: str,
    algo: str = "sac",
    retrain: bool = True,
    timesteps: int = None,
    start_fold: int = 0,
    with_sentiment: bool = False,
    seed: int = 42,
    device: str = "cpu",
    fold_indices: list[int] | None = None,
) -> dict:
    from backtest.runner import WalkForwardBacktester

    # Ablation: LLM sentiment channel. OFF (default) leaves the sentiment block
    # as zeros (same obs_dim, so directly comparable to the sentiment-off
    # baseline). ON fills it from the precomputed cache — noise+dropout on the
    # TRAIN fn (robustness, mirrors production), clean on the EVAL/test fn.
    sentiment_fn = sentiment_fn_eval = None
    if with_sentiment:
        from pathlib import Path as _Path
        from llm.sentiment_cache import SentimentCache
        cache_path = _Path("data/sentiment_cache.db")
        if not cache_path.exists():
            raise FileNotFoundError(
                "--with-sentiment requested but data/sentiment_cache.db missing. "
                "Run: python precompute_sentiment.py --model local"
            )
        cache   = SentimentCache(cache_path)
        vec_dim = len(config["market"]["tickers"]) * config["llm"]["sentiment_dim"]
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
            training=True,
            model=model_id,
            schema_version=schema_version,
            seed=seed,
        )
        sentiment_fn_eval = cache.make_lookup_fn(
            vector_dim=vec_dim, noise_std=0.0, dropout_prob=0.0, training=False,
            model=model_id, schema_version=schema_version, seed=seed,
        )
        config.setdefault("experiment", {}).update(
            {
                "sentiment_model": model_id,
                "sentiment_schema_version": schema_version,
            }
        )
        logger.info(
            "Sentiment ablation: ON | model=%s | dates=%d | "
            "nonzero=%.1f%% | vec_dim=%d",
            model_id,
            coverage["required_dates"],
            coverage["nonzero_date_rate"] * 100,
            vec_dim,
        )

    backtester = WalkForwardBacktester(
        config=config,
        risk_profile=risk_profile,
        term=term,
        algo=algo,
        timesteps_override=timesteps,
        start_fold=start_fold,
        sentiment_fn=sentiment_fn,
        sentiment_fn_eval=sentiment_fn_eval,
        seed=seed,
        device=device,
        fold_indices=fold_indices,
    )
    result = backtester.run(retrain_each_fold=retrain)
    return result.aggregate()


def print_summary_table(results: list[dict]) -> None:
    table = Table(title="latent.alpha — Backtest Summary", border_style="dim")
    table.add_column("Risk profile",     style="cyan")
    table.add_column("Term",             style="cyan")
    table.add_column("Mean return",      style="green")
    table.add_column("Mean Sharpe",      style="yellow")
    table.add_column("Med Sharpe",       style="yellow")
    table.add_column("Mean max DD",      style="red")
    table.add_column("Mean Calmar",      style="magenta")
    table.add_column("Med Calmar",       style="magenta")
    table.add_column("Folds",            style="dim")

    for r in results:
        table.add_row(
            r.get("risk_profile", "-"),
            r.get("term", "-"),
            f"{r.get('mean_annual_return_pct', 0):.1f}%",
            f"{r.get('mean_sharpe', 0):.2f}",
            f"{r.get('median_sharpe', 0):.2f}",
            f"{r.get('mean_max_drawdown_pct', 0):.1f}%",
            f"{r.get('mean_calmar', 0):.2f}",
            f"{r.get('median_calmar', 0):.2f}",
            str(r.get("n_folds", 0)),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="latent.alpha — walk-forward backtest")
    parser.add_argument("--algo",          default="sac",
                        choices=["sac", "ppo"],
                        help="RL algorithm (default: sac)")
    parser.add_argument("--risk",          default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term",          default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--all-profiles",  action="store_true",
                        help="Run all three risk profiles")
    parser.add_argument("--no-retrain",    action="store_true",
                        help="Skip retraining — use existing model checkpoints")
    parser.add_argument("--timesteps",    type=int, default=None,
                        help="Override training timesteps per fold (e.g. 1000000 for faster runs)")
    parser.add_argument("--start-fold",   type=int, default=0,
                        help="0-indexed fold to start running from. Earlier folds are reused "
                             "from persisted state if present, else skipped (durable resume).")
    parser.add_argument("--with-sentiment", action="store_true",
                        help="Fill the LLM sentiment channel from data/sentiment_cache.db "
                             "(ablation ON arm). Default OFF leaves it zeroed.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Training device (default: cpu)")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Specific outer fold indices to evaluate")
    args = parser.parse_args()

    config   = load_config()
    profiles = ["conservative", "balanced", "aggressive"] if args.all_profiles else [args.risk]

    all_results = []
    for profile in profiles:
        console.rule(f"[bold]{args.algo.upper()} / {profile.upper()} / {args.term.upper()}[/bold]")
        agg = run_backtest(
            config=config,
            risk_profile=profile,
            term=args.term,
            algo=args.algo,
            retrain=not args.no_retrain,
            timesteps=args.timesteps,
            start_fold=args.start_fold,
            with_sentiment=args.with_sentiment,
            seed=args.seed,
            device=args.device,
            fold_indices=args.folds,
        )
        agg["term"] = args.term
        all_results.append(agg)

    print_summary_table(all_results)
    console.print(f"\n[dim]Results and equity curves saved to:[/dim] {config['paths']['results_dir']}")


if __name__ == "__main__":
    main()

"""
experiments/runner.py
═════════════════════
Systematic experiment runner for latent.alpha.

Runs walk-forward backtests across multiple config variants and produces
a comparison table. Each experiment gets its own results subdirectory.

Usage
-----
# Run all experiments
python -m experiments.runner

# Run specific experiments
python -m experiments.runner --experiments A_baseline_revised B_arch_tuned

# Run with custom timesteps
python -m experiments.runner --timesteps 1000000

# List available experiments
python -m experiments.runner --list
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from rich.console import Console
from rich.table import Table

# Fix macOS ARM libomp conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

console = Console()
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
RESULTS_BASE = Path(__file__).parent.parent / "results" / "experiments"


def load_base_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base config. Overrides win on conflicts."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_single_experiment(
    name: str,
    config: dict,
    description: str,
    risk_profile: str = "balanced",
    term: str = "medium",
    timesteps_override: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    fold_indices: list[int] | None = None,
    results_base: Path | None = None,
) -> Dict:
    """
    Run a walk-forward backtest for one experiment configuration.

    Returns dict with experiment name, metrics, and timing.
    """
    from backtest.runner import WalkForwardBacktester

    console.rule(f"[bold cyan]{name}[/bold cyan]")
    console.print(f"[dim]{description}[/dim]")
    console.print(f"risk={risk_profile} | term={term}")

    # Set up experiment-specific results directory
    exp_results_dir = (results_base or RESULTS_BASE) / name
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    config["paths"]["results_dir"] = str(exp_results_dir)

    t0 = time.time()

    try:
        backtester = WalkForwardBacktester(
            config,
            risk_profile=risk_profile,
            term=term,
            timesteps_override=timesteps_override,
            seed=seed,
            device=device,
            fold_indices=fold_indices,
        )
        bt_result = backtester.run()

        elapsed = time.time() - t0

        # Extract aggregate metrics from BacktestResult
        agg = bt_result.aggregate()
        benchmark = agg.get("baselines", {}).get("benchmark", {})
        median_sharpe = agg.get("median_sharpe", 0)
        std_sharpe = agg.get("std_sharpe", 0)
        mean_drawdown = agg.get("mean_max_drawdown_pct", 0)
        benchmark_shortfall = max(
            benchmark.get("median_sharpe", 0) - median_sharpe,
            0,
        )
        robust_score = (
            median_sharpe
            - 0.25 * std_sharpe
            - 0.02 * mean_drawdown
            - 0.5 * benchmark_shortfall
        )
        metrics = {
            "experiment": name,
            "description": description,
            "risk_profile": risk_profile,
            "term": term,
            "seed": seed,
            "device": device,
            "run_id": backtester.run_id,
            "identity_sha256": backtester.identity_sha256,
            "n_folds": agg.get("n_folds", 0),
            "n_valid_folds": agg.get("n_valid_folds", 0),
            "mean_annual_return_pct": agg.get("mean_annual_return_pct", 0),
            "std_annual_return_pct": agg.get("std_annual_return_pct", 0),
            "mean_sharpe": agg.get("mean_sharpe", 0),
            "median_sharpe": median_sharpe,
            "std_sharpe": std_sharpe,
            "worst_fold_sharpe": agg.get("worst_fold_sharpe", 0),
            "mean_max_drawdown_pct": agg.get("mean_max_drawdown_pct", 0),
            "mean_calmar": agg.get("mean_calmar", 0),
            "robust_score": round(float(robust_score), 4),
            "baselines": agg.get("baselines", {}),
            "elapsed_hours": elapsed / 3600,
        }

        console.print(
            f"[green]✓ {name}[/green] — "
            f"Return: {metrics['mean_annual_return_pct']:+.1f}% | "
            f"Sharpe: {metrics['mean_sharpe']:.3f} | "
            f"MaxDD: {metrics['mean_max_drawdown_pct']:.1f}% | "
            f"Time: {elapsed/3600:.1f}h"
        )

        # Save individual experiment results
        with open(backtester.results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        elapsed = time.time() - t0
        console.print(f"[red]✗ {name} failed after {elapsed/60:.0f}m: {e}[/red]")
        logger.exception("Experiment %s failed", name)
        return {
            "experiment": name,
            "description": description,
            "error": str(e),
            "elapsed_hours": elapsed / 3600,
        }


def print_comparison_table(all_results: List[Dict]):
    """Print a formatted comparison table of all experiment results."""
    table = Table(title="Experiment Comparison — balanced / medium", show_lines=True)
    table.add_column("Experiment", style="cyan", width=22)
    table.add_column("Seed", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD %", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Std %", justify="right")
    table.add_column("Time (h)", justify="right")

    # Best tracking
    best_score = -999
    best_name = ""

    for r in all_results:
        if "error" in r:
            table.add_row(r["experiment"], str(r.get("seed", "—")), "[red]FAILED[/red]", "—", "—", "—", "—",
                          f"{r.get('elapsed_hours', 0):.1f}")
            continue

        ret = r["mean_annual_return_pct"]
        sharpe = r["median_sharpe"]
        dd = r["mean_max_drawdown_pct"]
        calmar = r["mean_calmar"]
        std = r["std_annual_return_pct"]
        hours = r["elapsed_hours"]

        # Highlight if beats v2
        ret_style = "green bold" if ret > 13.1 else ("green" if ret > 4.0 else "red")
        sharpe_style = "green bold" if sharpe > 0.94 else ("green" if sharpe > 0.446 else "red")

        score = r["robust_score"]
        if score > best_score:
            best_score = score
            best_name = r["experiment"]

        table.add_row(
            r["experiment"],
            str(r.get("seed", "—")),
            f"[{ret_style}]{ret:+.1f}[/{ret_style}]",
            f"[{sharpe_style}]{sharpe:.3f}[/{sharpe_style}]",
            f"{dd:.1f}",
            f"{calmar:.3f}",
            f"{std:.1f}",
            f"{hours:.1f}",
        )

    console.print()
    console.print(table)

    if best_name:
        console.print(
            f"\n[bold]Best robust score: {best_name} "
            f"({best_score:.3f})[/bold]"
        )


def save_comparison(all_results: List[Dict]):
    """Save comparison results to CSV and JSON."""
    import csv

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = RESULTS_BASE / f"comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    csv_path = RESULTS_BASE / f"comparison_{timestamp}.csv"
    if all_results:
        keys = [k for k in all_results[0].keys() if k != "error"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                if "error" not in r:
                    writer.writerow(r)

    console.print(f"\n[dim]Results saved to {csv_path}[/dim]")


def main():
    from experiments.configs import EXPERIMENTS, get_experiment_names

    parser = argparse.ArgumentParser(
        description="latent.alpha — systematic experiment runner",
    )
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment names to run (default: all)")
    parser.add_argument("--risk", default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term", default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override training timesteps per fold")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Replicate seeds (default: 42)")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Training device (default: cpu)")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Specific outer fold indices to evaluate")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit")
    args = parser.parse_args()

    if args.list:
        console.print("[bold]Available experiments:[/bold]\n")
        for name, exp in EXPERIMENTS.items():
            console.print(f"  [cyan]{name:25s}[/cyan] {exp['description']}")
        return

    # Select experiments
    exp_names = args.experiments or get_experiment_names()
    for name in exp_names:
        if name not in EXPERIMENTS:
            console.print(f"[red]Unknown experiment: {name}[/red]")
            console.print(f"Available: {get_experiment_names()}")
            return

    console.rule("[bold]latent.alpha — Experiment Suite[/bold]")
    console.print(
        f"Running {len(exp_names)} experiments × {len(args.seeds)} seed(s): "
        f"{', '.join(exp_names)}"
    )
    console.print(f"Risk: {args.risk} | Term: {args.term} | Timesteps: {args.timesteps or 'default (3M)'}")
    console.print()

    base_config = load_base_config()
    all_results = []

    for exp_name in exp_names:
        exp = EXPERIMENTS[exp_name]
        for seed in args.seeds:
            config = deep_merge(base_config, exp["overrides"])
            result = run_single_experiment(
                name=exp_name,
                config=config,
                description=exp["description"],
                risk_profile=args.risk,
                term=args.term,
                timesteps_override=args.timesteps,
                seed=seed,
                device=args.device,
                fold_indices=args.folds,
            )
            all_results.append(result)

    # Print comparison and save
    print_comparison_table(all_results)
    save_comparison(all_results)

    console.rule("[bold green]Experiment suite complete[/bold green]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""
experiments/hyperparam_sweep.py
═══════════════════════════════
Fast hyperparameter sweep using short training runs (1 fold, 200K steps).

Finds the best (reward_scaling, ent_coef, learning_rate) combination by
running a grid of configs and comparing eval Sharpe after each mini-run.
Each config takes ~3 hours instead of 40+, so a 12-config grid finishes
in ~36 hours sequentially, or you can cherry-pick 6 configs for ~18 hours.

Usage
-----
# Run full grid (all combos)
python -m experiments.hyperparam_sweep

# Run specific configs by index
python -m experiments.hyperparam_sweep --configs 0 1 2 3

# Custom timesteps (default 200K — enough to see learning signal)
python -m experiments.hyperparam_sweep --timesteps 300000

# List all configs without running
python -m experiments.hyperparam_sweep --list

"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

# Fix macOS ARM libomp conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

console = Console()
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
RESULTS_BASE = Path(__file__).parent.parent / "results" / "sweep"


# ── Hyperparameter grid ─────────────────────────────────────────────────────
# These are the parameters that most affect whether the critic learns.
# reward_scaling amplifies the signal; ent_coef controls exploration;
# learning_rate controls how fast the critic adapts.

SWEEP_GRID = {
    "reward_scaling": [1.0, 10.0, 50.0, 100.0],
    "ent_coef":       [0.01, 0.02, 0.05],
    "learning_rate":  [3e-4],            # Keep fixed for now — add [1e-4, 3e-4] if needed
}


def generate_sweep_configs() -> List[Dict]:
    """
    Generate all combinations from the sweep grid.

    Returns list of dicts, each with:
      - name: human-readable config name
      - overrides: config overrides to deep_merge
      - description: what this config tests
    """
    configs = []
    keys = sorted(SWEEP_GRID.keys())

    for combo in product(*[SWEEP_GRID[k] for k in keys]):
        params = dict(zip(keys, combo))

        name = (
            f"rs{params['reward_scaling']:.0f}_"
            f"ec{params['ent_coef']:.3f}_"
            f"lr{params['learning_rate']:.0e}"
        ).replace("+", "")

        overrides = {
            "env": {
                "reward_scaling": params["reward_scaling"],
                "fail_on_dead_channel": 0.5,
            },
            "sac": {
                "ent_coef": params["ent_coef"],
                "learning_rate": params["learning_rate"],
            },
        }

        description = (
            f"reward_scaling={params['reward_scaling']}, "
            f"ent_coef={params['ent_coef']}, "
            f"lr={params['learning_rate']}"
        )

        configs.append({
            "name": name,
            "overrides": overrides,
            "description": description,
            "params": params,
        })

    return configs


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base config."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_single_sweep(
    name: str,
    config: dict,
    description: str,
    timesteps: int = 200_000,
    risk_profile: str = "balanced",
    term: str = "medium",
    seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """
    Run a single short training + eval cycle.

    Forces exactly 1 walk-forward fold for speed.
    Returns dict with metrics and timing.
    """
    from backtest.runner import WalkForwardBacktester
    import pandas as pd

    console.rule(f"[bold cyan]{name}[/bold cyan]")
    console.print(f"[dim]{description}[/dim]")
    console.print(f"timesteps={timesteps:,} | risk={risk_profile} | term={term}")

    # Results directory for this config
    exp_dir = RESULTS_BASE / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config["paths"]["results_dir"] = str(exp_dir)

    # Force exactly 1 fold by constraining backtest end_date to the
    # first fold's test_end (start + train_months + test_months).
    config = copy.deepcopy(config)
    bt_cfg = config["backtest"]
    start_ts = pd.Timestamp(bt_cfg["start_date"])
    one_fold_end = (
        start_ts
        + pd.DateOffset(months=bt_cfg["walk_forward_train_months"])
        + pd.DateOffset(months=bt_cfg["walk_forward_test_months"])
    )
    bt_cfg["end_date"] = one_fold_end.strftime("%Y-%m-%d")

    t0 = time.time()

    try:
        backtester = WalkForwardBacktester(
            config,
            risk_profile=risk_profile,
            term=term,
            timesteps_override=timesteps,
            seed=seed,
            device=device,
        )
        folds = backtester.__class__._generate_folds(backtester)
        console.print(f"walk-forward folds for this run: {len(folds)}")
        bt_result = backtester.run()

        elapsed = time.time() - t0
        agg = bt_result.aggregate()

        metrics = {
            "name": name,
            "description": description,
            "timesteps": timesteps,
            "seed": seed,
            "device": device,
            "run_id": backtester.run_id,
            "identity_sha256": backtester.identity_sha256,
            "n_folds": agg.get("n_folds", 0),
            "n_valid_folds": agg.get("n_valid_folds", 0),
            "mean_annual_return_pct": agg.get("mean_annual_return_pct", 0),
            "std_annual_return_pct": agg.get("std_annual_return_pct", 0),
            "mean_sharpe": agg.get("mean_sharpe", 0),
            "mean_max_drawdown_pct": agg.get("mean_max_drawdown_pct", 0),
            "mean_calmar": agg.get("mean_calmar", 0),
            "elapsed_hours": elapsed / 3600,
        }

        # Style based on performance
        sharpe = metrics["mean_sharpe"]
        ret = metrics["mean_annual_return_pct"]
        style = "green bold" if sharpe > 0.5 else ("green" if sharpe > 0 else "red")

        console.print(
            f"[{style}]✓ {name}[/{style}] — "
            f"Return: {ret:+.1f}% | "
            f"Sharpe: {sharpe:.3f} | "
            f"MaxDD: {metrics['mean_max_drawdown_pct']:.1f}% | "
            f"Time: {elapsed/3600:.1f}h"
        )

        # Save metrics
        with open(backtester.results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        elapsed = time.time() - t0
        console.print(f"[red]✗ {name} failed after {elapsed/60:.0f}m: {e}[/red]")
        logger.exception("Sweep config %s failed", name)
        return {
            "name": name,
            "description": description,
            "error": str(e),
            "elapsed_hours": elapsed / 3600,
        }


def print_sweep_results(results: List[Dict]):
    """Print a ranked comparison table of all sweep results."""
    table = Table(
        title="Hyperparameter Sweep Results (ranked by Sharpe)",
        show_lines=True,
    )
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Config", style="cyan", width=28)
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD %", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Time (h)", justify="right")

    # Sort by Sharpe descending (filter out errors)
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    valid.sort(key=lambda r: r["mean_sharpe"], reverse=True)

    for rank, r in enumerate(valid, 1):
        sharpe = r["mean_sharpe"]
        ret = r["mean_annual_return_pct"]

        sharpe_style = "green bold" if sharpe > 0.5 else ("green" if sharpe > 0 else "red")
        ret_style = "green" if ret > 0 else "red"
        # Empty rank_style would produce "[]N[/]" which breaks Rich markup.
        rank_cell = f"[bold]{rank}[/bold]" if rank <= 3 else str(rank)

        table.add_row(
            rank_cell,
            escape(str(r["name"])),
            f"[{ret_style}]{ret:+.1f}[/{ret_style}]",
            f"[{sharpe_style}]{sharpe:.3f}[/{sharpe_style}]",
            f"{r['mean_max_drawdown_pct']:.1f}",
            f"{r['mean_calmar']:.3f}",
            f"{r['elapsed_hours']:.1f}",
        )

    for r in errors:
        table.add_row(
            "—",
            escape(str(r["name"])),
            "[red]FAILED[/red]",
            "—",
            "—",
            "—",
            f"{r.get('elapsed_hours', 0):.1f}",
        )

    console.print()
    console.print(table)

    if valid:
        best = valid[0]
        console.print(f"\n[bold green]Best config: {escape(str(best['name']))}[/bold green]")
        console.print(f"  Sharpe: {best['mean_sharpe']:.3f} | Return: {best['mean_annual_return_pct']:+.1f}%")
        console.print(
            "  → Apply with: [cyan]python -m experiments.hyperparam_sweep --apply "
            f"{escape(str(best['name']))}[/cyan]"
        )


def save_sweep_results(results: List[Dict]):
    """Save sweep results to JSON and CSV."""
    import csv

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = RESULTS_BASE / f"sweep_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = RESULTS_BASE / f"sweep_{timestamp}.csv"
    valid = [r for r in results if "error" not in r]
    if valid:
        keys = [k for k in valid[0].keys() if k != "error"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in valid:
                writer.writerow(r)

    console.print(f"\n[dim]Results saved to {json_path}[/dim]")


def apply_best_config(name: str):
    """Apply a sweep winner's params to config.yaml."""
    metrics_path = RESULTS_BASE / name / "metrics.json"
    if not metrics_path.exists():
        console.print(f"[red]No results found for {name}[/red]")
        return

    # Parse params from the config name
    # Format: rs{reward_scaling}_ec{ent_coef}_lr{learning_rate}
    parts = name.split("_")
    params = {}
    for p in parts:
        if p.startswith("rs"):
            params["reward_scaling"] = float(p[2:])
        elif p.startswith("ec"):
            params["ent_coef"] = float(p[2:])
        elif p.startswith("lr"):
            params["learning_rate"] = float(p[2:])

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    console.print(f"[bold]Applying sweep winner: {name}[/bold]")
    console.print(f"  reward_scaling: {config['env'].get('reward_scaling')} → {params.get('reward_scaling')}")
    console.print(f"  ent_coef:       {config['sac'].get('ent_coef')} → {params.get('ent_coef')}")
    console.print(f"  learning_rate:  {config['sac'].get('learning_rate')} → {params.get('learning_rate')}")

    if "reward_scaling" in params:
        config["env"]["reward_scaling"] = params["reward_scaling"]
    if "ent_coef" in params:
        config["sac"]["ent_coef"] = params["ent_coef"]
    if "learning_rate" in params:
        config["sac"]["learning_rate"] = params["learning_rate"]

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print("[green]✓ config.yaml updated[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="latent.alpha — hyperparameter sweep (short runs)",
    )
    parser.add_argument("--configs", nargs="+", type=int, default=None,
                        help="Config indices to run (default: all)")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Training timesteps per fold (default: 200K)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Replicate seeds (default: 42)")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Training device (default: cpu)")
    parser.add_argument("--risk", default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--term", default="medium",
                        choices=["short", "medium", "long"])
    parser.add_argument("--list", action="store_true",
                        help="List all configs and exit")
    parser.add_argument("--apply", type=str, default=None,
                        help="Apply a sweep winner's params to config.yaml")
    args = parser.parse_args()

    if args.apply:
        apply_best_config(args.apply)
        return

    sweep_configs = generate_sweep_configs()

    if args.list:
        console.print("[bold]Sweep grid:[/bold]\n")
        for i, cfg in enumerate(sweep_configs):
            console.print(f"  [{i:2d}] [cyan]{cfg['name']:30s}[/cyan] {cfg['description']}")
        console.print(f"\n  Total: {len(sweep_configs)} configs")
        console.print("  Fold policy: forced to 1 fold per config")
        console.print(f"  Est. time at {args.timesteps:,} steps: ~{len(sweep_configs) * 2.5:.0f}h")
        console.print(f"  Est. time for 4 configs at {args.timesteps:,} steps: ~{4 * args.timesteps / 200_000 * 2.5:.0f}h")
        console.print(f"\n  Run a subset: python -m experiments.hyperparam_sweep --configs 0 3 6 9")
        return

    # Select configs to run
    if args.configs is not None:
        selected = [sweep_configs[i] for i in args.configs if i < len(sweep_configs)]
    else:
        selected = sweep_configs

    console.rule("[bold]latent.alpha — Hyperparameter Sweep[/bold]")
    console.print(
        f"Running {len(selected)} configs × {len(args.seeds)} seed(s) "
        f"× {args.timesteps:,} steps each"
    )
    console.print("Backtest mode: forced to exactly 1 fold per config")
    console.print(f"Grid: {SWEEP_GRID}")
    console.print(f"Est. time: ~{len(selected) * args.timesteps / 200_000 * 2.5:.0f}h\n")

    # Load base config
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    all_results = []

    for cfg in selected:
        for seed in args.seeds:
            config = deep_merge(base_config, cfg["overrides"])
            result = run_single_sweep(
                name=cfg["name"],
                config=config,
                description=cfg["description"],
                timesteps=args.timesteps,
                risk_profile=args.risk,
                term=args.term,
                seed=seed,
                device=args.device,
            )
            all_results.append(result)

    # Persist first so a Rich/console failure cannot lose sweep aggregates.
    save_sweep_results(all_results)
    print_sweep_results(all_results)

    console.rule("[bold green]Sweep complete[/bold green]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""Freeze a selected system and evaluate only on untouched lockbox folds."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from experiments.configs import EXPERIMENTS
from experiments.runner import (
    deep_merge,
    load_base_config,
    run_single_experiment,
)
from experiments.successive_halving import hyperparameter_candidates


logger = logging.getLogger(__name__)


def _load_winner(path: Path) -> str:
    with open(path) as f:
        state = json.load(f)
    winner = state.get("winner")
    if not winner:
        raise ValueError(f"Search at {path} has no frozen winner.")
    return winner


def run_final_validation(
    *,
    architecture_state: Path,
    hyperparameter_state: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    architecture = _load_winner(architecture_state)
    hyperparameter = _load_winner(hyperparameter_state)
    if architecture not in EXPERIMENTS:
        raise ValueError(f"Unknown architecture winner: {architecture}")
    if hyperparameter not in hyperparameter_candidates():
        raise ValueError(f"Unknown hyperparameter winner: {hyperparameter}")

    config = load_base_config()
    config = deep_merge(
        config,
        EXPERIMENTS[architecture]["overrides"],
    )
    config = deep_merge(
        config,
        hyperparameter_candidates()[hyperparameter],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_config.yaml"
    with open(frozen_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    folds = list(config["backtest"]["lockbox_folds"])
    seeds = (42, 123, 777)
    scenarios = {
        "base": {},
        "cost_1_5x": {
            "env": {
                "transaction_cost_pct": (
                    config["env"]["transaction_cost_pct"] * 1.5
                ),
                "slippage_pct": config["env"]["slippage_pct"] * 1.5,
            }
        },
        "cost_2x_delay_2": {
            "env": {
                "transaction_cost_pct": (
                    config["env"]["transaction_cost_pct"] * 2.0
                ),
                "slippage_pct": config["env"]["slippage_pct"] * 2.0,
                "execution_lag_bars": 2,
            }
        },
    }

    state_path = output_dir / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
    else:
        state = {"tasks": {}}

    for scenario, overrides in scenarios.items():
        for seed in seeds:
            task = f"{scenario}:s{seed}"
            if state["tasks"].get(task, {}).get("status") == "completed":
                continue
            scenario_config = deep_merge(config, overrides)
            result = run_single_experiment(
                name=(
                    f"lockbox_{architecture}_{hyperparameter}_"
                    f"{scenario}_s{seed}"
                ),
                config=scenario_config,
                description=(
                    f"Frozen {architecture}/{hyperparameter}; {scenario}"
                ),
                timesteps_override=200_000,
                seed=seed,
                device=device,
                fold_indices=folds,
                results_base=output_dir / "runs",
            )
            state["tasks"][task] = {
                "status": (
                    "failed" if "error" in result else "completed"
                ),
                "result": result,
            }
            temporary = state_path.with_suffix(".tmp")
            with open(temporary, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            temporary.replace(state_path)

    summary = {
        "architecture": architecture,
        "hyperparameter": hyperparameter,
        "lockbox_folds": folds,
        "scenarios": {},
    }
    for scenario in scenarios:
        results = [
            state["tasks"][f"{scenario}:s{seed}"]["result"]
            for seed in seeds
            if state["tasks"].get(f"{scenario}:s{seed}", {}).get("status")
            == "completed"
        ]
        scores = [row["robust_score"] for row in results]
        sharpes = [row["median_sharpe"] for row in results]
        returns = [row["mean_annual_return_pct"] for row in results]
        summary["scenarios"][scenario] = {
            "replicates": len(results),
            "median_robust_score": (
                float(np.median(scores)) if scores else None
            ),
            "median_sharpe": (
                float(np.median(sharpes)) if sharpes else None
            ),
            "median_annual_return_pct": (
                float(np.median(returns)) if returns else None
            ),
            "worst_seed_sharpe": (
                float(np.min(sharpes)) if sharpes else None
            ),
        }
    base = summary["scenarios"]["base"]
    severe = summary["scenarios"]["cost_2x_delay_2"]
    summary["promotion_passed"] = bool(
        base["replicates"] == len(seeds)
        and base["median_robust_score"] is not None
        and base["median_robust_score"] > 0
        and severe["median_sharpe"] is not None
        and severe["median_sharpe"] > 0
    )
    with open(output_dir / "final_report.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Final validation: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen lockbox and stress validation"
    )
    parser.add_argument("--architecture-state", required=True, type=Path)
    parser.add_argument("--hyperparameter-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_final_validation(
        architecture_state=args.architecture_state,
        hyperparameter_state=args.hyperparameter_state,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

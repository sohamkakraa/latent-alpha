"""Persistent multi-seed, multi-fold successive-halving search."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable

import numpy as np

from experiments.configs import EXPERIMENTS
from experiments.runner import (
    deep_merge,
    load_base_config,
    run_single_experiment,
)


logger = logging.getLogger(__name__)
SEARCH_ROOT = Path("results/search")


@dataclass(frozen=True)
class Stage:
    name: str
    timesteps: int
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    keep: int


ARCHITECTURE_STAGES = (
    Stage("screen", 25_000, (0,), (42,), 4),
    Stage("regimes", 75_000, (0, 3, 6), (42, 123), 2),
    Stage(
        "development",
        200_000,
        (0, 1, 2, 3, 4, 5, 6),
        (42, 123, 777),
        1,
    ),
)

HYPERPARAMETER_STAGES = (
    Stage("screen", 25_000, (1, 5), (42,), 6),
    Stage("regimes", 75_000, (0, 3, 6), (42, 123), 3),
    Stage(
        "development",
        200_000,
        (0, 1, 2, 3, 4, 5, 6),
        (42, 123, 777),
        1,
    ),
)


def hyperparameter_candidates() -> Dict[str, dict]:
    """Focused, non-cartesian candidates around prior useful settings."""
    rows = [
        ("base", 3e-4, 0.01, 10.0, 50.0, 0.05, [512, 512]),
        ("low_lr", 1e-4, 0.01, 10.0, 50.0, 0.05, [512, 512]),
        ("high_lr", 5e-4, 0.01, 10.0, 50.0, 0.05, [512, 512]),
        ("low_entropy", 3e-4, 0.005, 10.0, 50.0, 0.05, [512, 512]),
        ("high_entropy", 3e-4, 0.02, 10.0, 50.0, 0.05, [512, 512]),
        ("low_turnover_penalty", 3e-4, 0.01, 10.0, 25.0, 0.05, [512, 512]),
        ("high_turnover_penalty", 3e-4, 0.01, 10.0, 100.0, 0.05, [512, 512]),
        ("small_delta", 3e-4, 0.01, 10.0, 50.0, 0.025, [512, 512]),
        ("large_delta", 3e-4, 0.01, 10.0, 50.0, 0.075, [512, 512]),
        ("small_network", 3e-4, 0.01, 10.0, 50.0, 0.05, [256, 256]),
        ("reward_scale_5", 3e-4, 0.01, 5.0, 50.0, 0.05, [512, 512]),
        ("reward_scale_20", 3e-4, 0.01, 20.0, 50.0, 0.05, [512, 512]),
    ]
    candidates = {}
    for name, lr, entropy, scale, turnover, delta, network in rows:
        candidates[name] = {
            "sac": {
                "learning_rate": lr,
                "ent_coef": entropy,
                "net_arch": network,
            },
            "env": {
                "reward_scaling": scale,
                "turnover_penalty": turnover,
                "max_weight_delta": delta,
            },
        }
    return candidates


def _load_state(path: Path, mode: str) -> dict:
    if path.exists():
        with open(path) as f:
            state = json.load(f)
        if state.get("mode") != mode:
            raise ValueError(
                f"Search state mode={state.get('mode')} does not match {mode}."
            )
        return state
    return {
        "schema_version": 1,
        "mode": mode,
        "tasks": {},
        "stage_rankings": {},
        "winner": None,
    }


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    temporary.replace(path)


def _candidate_score(results: Iterable[dict]) -> float:
    rows = [row for row in results if "error" not in row]
    if not rows:
        return -1_000.0
    scores = np.asarray(
        [row.get("robust_score", -1_000.0) for row in rows],
        dtype=float,
    )
    # Median performance penalized by seed/fold instability.
    return float(np.median(scores) - 0.25 * np.std(scores))


def run_search(
    *,
    mode: str,
    search_name: str,
    architecture: str,
    device: str,
) -> dict:
    state_path = SEARCH_ROOT / search_name / "state.json"
    state = _load_state(state_path, mode)
    base = load_base_config()
    base["paths"]["results_dir"] = str(
        SEARCH_ROOT / search_name / "runs"
    )

    if mode == "architecture":
        candidates = {
            name: experiment["overrides"]
            for name, experiment in EXPERIMENTS.items()
        }
        descriptions = {
            name: experiment["description"]
            for name, experiment in EXPERIMENTS.items()
        }
        stages = ARCHITECTURE_STAGES
    else:
        if architecture not in EXPERIMENTS:
            raise ValueError(
                f"Unknown architecture {architecture}; "
                f"choose from {sorted(EXPERIMENTS)}"
            )
        architecture_overrides = EXPERIMENTS[architecture]["overrides"]
        candidates = {
            name: deep_merge(architecture_overrides, overrides)
            for name, overrides in hyperparameter_candidates().items()
        }
        descriptions = {
            name: f"{architecture} with hyperparameter candidate {name}"
            for name in candidates
        }
        stages = HYPERPARAMETER_STAGES

    survivors = list(candidates)
    for stage in stages:
        stage_results: Dict[str, list[dict]] = {
            candidate: [] for candidate in survivors
        }
        for candidate in survivors:
            for seed in stage.seeds:
                task_key = (
                    f"{stage.name}:{candidate}:s{seed}:"
                    f"t{stage.timesteps}:f{','.join(map(str, stage.folds))}"
                )
                existing = state["tasks"].get(task_key)
                if existing and existing.get("status") == "completed":
                    result = existing["result"]
                else:
                    config = deep_merge(base, candidates[candidate])
                    config["sac"]["learning_starts"] = min(
                        int(config["sac"]["learning_starts"]),
                        max(1_000, stage.timesteps // 10),
                    )
                    config["sac"]["buffer_size"] = min(
                        int(config["sac"]["buffer_size"]),
                        max(50_000, stage.timesteps),
                    )
                    state["tasks"][task_key] = {"status": "running"}
                    _save_state(state_path, state)
                    result = run_single_experiment(
                        name=(
                            f"{search_name}_{stage.name}_{candidate}_s{seed}"
                        ),
                        config=config,
                        description=descriptions[candidate],
                        timesteps_override=stage.timesteps,
                        seed=seed,
                        device=device,
                        fold_indices=list(stage.folds),
                        results_base=SEARCH_ROOT / search_name / "runs",
                    )
                    state["tasks"][task_key] = {
                        "status": (
                            "failed" if "error" in result else "completed"
                        ),
                        "result": result,
                    }
                    _save_state(state_path, state)
                stage_results[candidate].append(result)

        ranking = sorted(
            (
                {
                    "candidate": candidate,
                    "score": round(
                        _candidate_score(stage_results[candidate]), 6
                    ),
                    "replicates": len(stage_results[candidate]),
                }
                for candidate in survivors
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        state["stage_rankings"][stage.name] = ranking
        survivors = [
            row["candidate"] for row in ranking[: stage.keep]
        ]
        _save_state(state_path, state)

    state["winner"] = survivors[0]
    _save_state(state_path, state)
    logger.info("Search winner: %s", state["winner"])
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent successive-halving optimizer"
    )
    parser.add_argument(
        "--mode",
        choices=["architecture", "hyperparameters"],
        required=True,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--architecture",
        default="lgbm_only",
        help="Architecture to tune in hyperparameters mode",
    )
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
    run_search(
        mode=args.mode,
        search_name=args.name,
        architecture=args.architecture,
        device=args.device,
    )


if __name__ == "__main__":
    main()

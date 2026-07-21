"""Tests for immutable experiment identity and run manifests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from backtest.runner import WalkForwardBacktester
from experiments.provenance import (
    build_run_identity,
    scientific_config,
    sha256_json,
)


def _config(tmp_path):
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    config["paths"]["results_dir"] = str(tmp_path / "results")
    config["paths"]["model_dir"] = str(tmp_path / "models")
    config["paths"]["logs_dir"] = str(tmp_path / "logs")
    return config


def test_scientific_hash_ignores_output_paths(tmp_path):
    first = _config(tmp_path)
    second = _config(tmp_path)
    second["paths"]["results_dir"] = "/different/output/location"
    second["paths"]["model_dir"] = "/different/model/location"

    assert sha256_json(scientific_config(first)) == sha256_json(
        scientific_config(second)
    )


def test_run_identity_changes_with_seed(tmp_path):
    config = _config(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    first, _ = build_run_identity(
        config,
        risk_profile="balanced",
        term="medium",
        algorithm="sac",
        seed=1,
        timesteps=100,
        device="cpu",
        repo_root=repo_root,
    )
    second, _ = build_run_identity(
        config,
        risk_profile="balanced",
        term="medium",
        algorithm="sac",
        seed=2,
        timesteps=100,
        device="cpu",
        repo_root=repo_root,
    )
    assert first != second


def test_backtester_isolates_runs_and_writes_manifest(tmp_path):
    config = _config(tmp_path)
    first = WalkForwardBacktester(
        config,
        seed=11,
        timesteps_override=100,
    )
    second = WalkForwardBacktester(
        config,
        seed=12,
        timesteps_override=100,
    )

    assert first.results_dir != second.results_dir
    assert first.config["paths"]["model_dir"] != second.config["paths"]["model_dir"]
    for backtester in (first, second):
        manifest_path = backtester.results_dir / "manifest.json"
        resolved_path = backtester.results_dir / "resolved_config.yaml"
        assert manifest_path.exists()
        assert resolved_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["run_id"] == backtester.run_id
        assert manifest["spec"]["seed"] == backtester.seed

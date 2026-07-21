"""Deterministic run identity and reproducibility manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


TRACKED_PACKAGES = (
    "gymnasium",
    "stable-baselines3",
    "torch",
    "pandas",
    "numpy",
    "lightgbm",
    "scipy",
    "yfinance",
)
CODE_PATHS = (
    ":(glob)**/*.py",
    ":(glob)**/*.toml",
    ":(glob)**/*.yaml",
    ":(glob)**/*.yml",
    ":(glob)**/*.sh",
    "README.md",
    "requirements.txt",
    ".gitignore",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def scientific_config(config: dict) -> dict:
    """Return config without machine/output paths that do not affect results."""
    payload = json.loads(_canonical(config))
    paths = payload.get("paths", {})
    for key in ("model_dir", "results_dir", "logs_dir"):
        paths.pop(key, None)
    return payload


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def code_state(repo_root: Path) -> Dict[str, Any]:
    status = _git(
        repo_root,
        "status",
        "--short",
        "--",
        *CODE_PATHS,
    )
    diff = _git(
        repo_root,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *CODE_PATHS,
    )
    untracked_digest = hashlib.sha256()
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = repo_root / relative
        if path.is_file():
            untracked_digest.update(relative.encode("utf-8"))
            untracked_digest.update(path.read_bytes())
    combined = (
        diff.encode("utf-8")
        + untracked_digest.hexdigest().encode("ascii")
    )
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD") or "unknown",
        "branch": _git(repo_root, "branch", "--show-current") or "unknown",
        "dirty": bool(status),
        "status": status.splitlines(),
        "worktree_diff_sha256": hashlib.sha256(combined).hexdigest(),
    }


def dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def build_run_identity(
    config: dict,
    *,
    risk_profile: str,
    term: str,
    algorithm: str,
    seed: int,
    timesteps: int | None,
    device: str,
    repo_root: Path,
) -> tuple[str, dict]:
    git = code_state(repo_root)
    config_payload = scientific_config(config)
    spec = {
        "risk_profile": risk_profile,
        "term": term,
        "algorithm": algorithm,
        "seed": int(seed),
        "timesteps": timesteps,
        "device": device,
    }
    identity_payload = {
        "config": config_payload,
        "spec": spec,
        "commit": git["commit"],
        "worktree_diff_sha256": git["worktree_diff_sha256"],
    }
    identity_hash = sha256_json(identity_payload)
    run_id = (
        f"{algorithm}_{risk_profile}_{term}_s{seed}_"
        f"{identity_hash[:12]}"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "identity_sha256": identity_hash,
        "config_sha256": sha256_json(config_payload),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": spec,
        "git": git,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": dependency_versions(),
        },
    }
    return run_id, manifest


def write_run_manifest(run_dir: Path, manifest: dict, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

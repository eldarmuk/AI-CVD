"""Deterministic synthetic artifacts and exact local provenance verification."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .baselines import predict
from .contracts import finite_number
from .evaluation import evaluate
from .synthetic import synthetic_episodes

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STUDY_ROOT.parent.parent
DEFAULT_CONFIG = STUDY_ROOT / "configs" / "synthetic_demo.toml"
ARTIFACT_NAMES = {"episodes.json", "predictions.json", "results.json", "config.toml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    expected = {
        "schema_version",
        "purpose",
        "change_threshold_bpm",
        "minimum_observations_per_window",
        "methods",
    }
    if set(config) != expected:
        raise ValueError("synthetic configuration must contain exactly the documented fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    if config["purpose"] != "synthetic_contract_verification":
        raise ValueError("this runner supports synthetic verification only")
    finite_number(config["change_threshold_bpm"], "change threshold", positive=True)
    minimum = config["minimum_observations_per_window"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum observations must be a positive integer")
    methods = config["methods"]
    if not isinstance(methods, list) or not methods or any(not isinstance(x, str) for x in methods):
        raise ValueError("methods must be a nonempty string list")
    if len(methods) != len(set(methods)) or set(methods) - {"last_observation", "window_median"}:
        raise ValueError("methods must be unique supported baselines")
    return config


def _payloads(config: dict) -> dict[str, dict]:
    episodes = synthetic_episodes()
    predictions = {
        method: [
            predict(episode.model_input(), method, config["minimum_observations_per_window"])
            for episode in episodes
        ]
        for method in config["methods"]
    }
    return {
        "episodes.json": {
            "synthetic_only": True,
            "episodes": [episode.to_dict() for episode in episodes],
        },
        "predictions.json": {
            "synthetic_only": True,
            "methods": {
                method: [asdict(prediction) for prediction in values]
                for method, values in predictions.items()
            },
        },
        "results.json": {
            "synthetic_only": True,
            "purpose": "arithmetic_and_contract_verification_not_scientific_performance",
            "methods": {
                method: evaluate(episodes, values, config["change_threshold_bpm"])
                for method, values in predictions.items()
            },
        },
    }


def _source_hashes(
    *, study_root: Path = STUDY_ROOT, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, str]:
    study_root, repository_root = study_root.resolve(), repository_root.resolve()
    if not study_root.is_relative_to(repository_root):
        raise ValueError("study source directory escaped its repository")
    files: set[Path] = set()
    for directory in ("hr_change", "scripts", "tests"):
        files.update((study_root / directory).rglob("*.py"))
    files.update((study_root / "configs").rglob("*.toml"))
    files.update(study_root.glob("requirements*.txt"))
    files.update(study_root.glob("*.lock"))
    if (study_root / "pyproject.toml").exists():
        files.add(study_root / "pyproject.toml")
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(repository_root):
            raise ValueError("source snapshot contains a link outside its declared files")
    return {path.relative_to(repository_root).as_posix(): sha256(path) for path in sorted(files)}


def _git_snapshot() -> dict:
    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1")
    return {
        "revision": revision,
        "worktree_dirty": bool(status) if status is not None else None,
        "worktree_scope": "entire_repository_including_untracked_files",
    }


def run_synthetic(output: Path, config_path: Path = DEFAULT_CONFIG) -> dict:
    output = output.resolve()
    if output.is_relative_to(STUDY_ROOT.resolve()):
        raise ValueError(
            "write generated artifacts outside the tracked study directory, for example under runs/"
        )
    config = read_config(config_path)
    payloads = _payloads(config)
    # Refuse overwrite even when the destination is empty. Partial failed runs
    # stay available for diagnosis; callers choose a new directory to retry.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        _write_json(output / name, payload)
    (output / "config.toml").write_bytes(config_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "synthetic_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_version": "invented_contract_examples_v1",
        "artifacts_sha256": {name: sha256(output / name) for name in sorted(ARTIFACT_NAMES)},
        "sources_sha256": _source_hashes(),
        "git": _git_snapshot(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "dependencies": "Python standard library only",
        },
    }
    _write_json(output / "manifest.json", manifest)
    verify_run(output)
    return manifest


def verify_run(output: Path) -> dict:
    output = output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(output):
        raise ValueError("manifest must be a regular file inside the run directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("synthetic_only") is not True
    ):
        raise ValueError("not a supported synthetic run manifest")
    if manifest.get("fixture_version") != "invented_contract_examples_v1":
        raise ValueError("unsupported synthetic fixture version")
    artifacts = manifest.get("artifacts_sha256", {})
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACT_NAMES:
        raise ValueError("unexpected or incomplete artifact inventory")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid SHA-256 digest: {name}")
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("artifact must be a regular file inside the run directory")
        if sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    if (
        not isinstance(manifest.get("sources_sha256"), dict)
        or manifest["sources_sha256"] != _source_hashes()
    ):
        raise ValueError("source snapshot differs; verify with the recorded source version")
    config = read_config(output / "config.toml")
    expected_payloads = _payloads(config)
    for name, expected_payload in expected_payloads.items():
        actual = json.loads((output / name).read_text(encoding="utf-8"))
        if actual != expected_payload:
            raise ValueError(f"semantic replay mismatch: {name}")
    return {
        "verified": True,
        "synthetic_only": True,
        "verification_scope": "artifact_integrity_and_current_source_semantic_replay",
        "git_revision_claim_verified": False,
        "artifacts_checked": len(artifacts),
        "sources_checked": len(manifest["sources_sha256"]),
    }

"""Run manifest writer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from subprocess import CalledProcessError


def _run_git_command(args: list[str]) -> str | None:
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [git_bin, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (CalledProcessError, OSError):
        return None
    return result.stdout.strip()


def _git_commit_hash() -> str | None:
    return _run_git_command(["rev-parse", "HEAD"])


def _is_git_dirty() -> bool | None:
    status = _run_git_command(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def _package_versions(packages: dict[str, str]) -> dict[str, str]:
    versions = {}
    for alias, package in packages.items():
        try:
            versions[alias] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[alias] = "unknown"
    return versions


def write_manifest(
    output_dir: Path,
    *,
    assay_format: str,
    resolved_config: dict,
    qc: dict[str, int],
    assay_counts: dict[str, int],
    filter_stats: dict[str, float],
    eb_summary: dict[str, object],
    split_summary: dict[str, object],
    export_counts: dict[str, int],
    thresholds: dict[str, float],
    split_seed: int,
) -> None:
    """Persist a JSON manifest with reproducibility metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "assay_format": assay_format,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "package_versions": _package_versions(
            {
                "polars": "polars",
                "numpy": "numpy",
                "matplotlib": "matplotlib",
                "hydra_core": "hydra-core",
                "rdkit": "rdkit",
            }
        ),
        "git": {
            "commit": _git_commit_hash(),
            "is_dirty": _is_git_dirty(),
        },
        "qc": qc,
        "assays": assay_counts,
        "filters": filter_stats,
        "empirical_bayes": {
            "total_compounds": eb_summary.get("total_compounds"),
            "retained_compounds": eb_summary.get("retained_compounds"),
            "passes_reliability_filter": eb_summary.get("passes_flagged"),
            "prior_mode": eb_summary.get("prior_mode"),
            "prior_alpha": eb_summary.get("prior_alpha"),
            "prior_beta": eb_summary.get("prior_beta"),
            "prior_mean_hit_rate": eb_summary.get("prior_mean_hit_rate"),
            "min_screens_for_prior_fit": eb_summary.get("min_screens_for_prior_fit"),
        },
        "export_counts": export_counts,
        "threshold_percentiles": thresholds,
        "split": {
            "seed": split_seed,
            "fractions": split_summary.get("fractions"),
            "counts": split_summary.get("split_counts"),
            "total_compounds": split_summary.get("total_compounds"),
            "unique_scaffolds_total": split_summary.get("unique_scaffolds_total"),
        },
        "config": resolved_config,
    }

    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

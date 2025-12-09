from __future__ import annotations

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import typer

from .datasets import (
    resolve_thresholds,
    write_regression_dataset,
    write_threshold_classifier_dataset,
    write_trimmed_dataset,
)
from .jobgen import ConfigError, JobDefinition, load_submission_plan

LOGGER = logging.getLogger("model_jobs_uge.cli")
app = typer.Typer(
    no_args_is_help=True,
    help="Generate UGE-ready Chemprop scripts and optionally submit them with qsub.",
)


class ScreensWeightMode(str, Enum):
    none = "none"
    linear = "linear"
    sqrt = "sqrt"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _write_scripts_and_maybe_submit(
    *,
    jobs: Sequence[JobDefinition],
    output_dir: Path,
    submit_override: bool | None,
    dry_run: bool,
    submitter: Callable[[Path], None],
    submit_label: str,
    logger: logging.Logger = LOGGER,
) -> None:
    for job in jobs:
        script_path = output_dir / job.filename
        content = job.content if job.content.endswith("\n") else f"{job.content}\n"
        script_path.write_text(content)
        os.chmod(script_path, 0o750)

        should_submit = job.submit if submit_override is None else submit_override

        if dry_run:
            logger.info("[DRY-RUN] Generated %s (%s skipped)", script_path, submit_label)
        elif should_submit:
            submitter(script_path)
        else:
            logger.info("Generated %s (%s skipped by config)", script_path, submit_label)


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    _setup_logging(verbose)


@app.command("prepare-regression")
def prepare_regression(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output Parquet file."),
    split_seed: int = typer.Option(..., "--split-seed", help="Data split seed."),
    compound_min_screens: float | None = typer.Option(
        None,
        "--compound-min-screens",
        help="Filter out compounds with fewer screens than this threshold.",
    ),
    compound_screens_column: str = typer.Option(
        "screens",
        "--compound-screens-column",
        help="Column holding the per-compound screen count.",
    ),
    screens_weight_mode: ScreensWeightMode = typer.Option(
        ScreensWeightMode.none,
        "--screens-weight-mode",
        help="Sample weight strategy: none, linear, or sqrt of the screens column.",
        case_sensitive=False,
        show_choices=True,
        metavar="none|linear|sqrt",
    ),
) -> None:
    """Build a regression-ready Parquet file with SMILES, split, and target columns."""
    split_column = f"split_seed{split_seed}"
    target_column = f"score_seed{split_seed}"
    smiles_column = "smiles"
    path = write_regression_dataset(
        input_path=input_path,
        output_path=output_path,
        smiles_column=smiles_column,
        split_column=split_column,
        target_column=target_column,
        compound_min_screens=compound_min_screens,
        compound_screens_column=compound_screens_column,
        screens_weight_mode=screens_weight_mode.value,
    )
    LOGGER.info("Wrote %s", path)
    typer.echo(path)


@app.command("prepare-threshold-classifier")
def prepare_threshold_classifier(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output Parquet file."),
    split_seed: int = typer.Option(..., "--split-seed", help="Data split seed."),
    lower_threshold: float | None = typer.Option(
        None, "--lower-threshold", help="Absolute lower threshold."
    ),
    upper_threshold: float | None = typer.Option(
        None, "--upper-threshold", help="Absolute upper threshold."
    ),
    thresholds_json: Path | None = typer.Option(
        None,
        "--thresholds-json",
        help="JSON file mapping percentiles to threshold values.",
    ),
    lower_percentile: float | None = typer.Option(
        None,
        "--lower-percentile",
        help="Percentile to use as the lower threshold (requires --thresholds-json).",
    ),
    upper_percentile: float | None = typer.Option(
        None,
        "--upper-percentile",
        help="Percentile to use as the upper threshold (requires --thresholds-json).",
    ),
    compound_min_screens: float | None = typer.Option(
        None,
        "--compound-min-screens",
        help="Filter out compounds with fewer screens than this threshold.",
    ),
    compound_screens_column: str = typer.Option(
        "screens",
        "--compound-screens-column",
        help="Column holding the per-compound screen count.",
    ),
    screens_weight_mode: ScreensWeightMode = typer.Option(
        ScreensWeightMode.none,
        "--screens-weight-mode",
        help="Sample weight strategy: none, linear, or sqrt of the screens column.",
        case_sensitive=False,
        show_choices=True,
        metavar="none|linear|sqrt",
    ),
) -> None:
    """Filter a dataset to binary labels and emit a classification Parquet file."""
    split_column = f"split_seed{split_seed}"
    metric_column = f"score_seed{split_seed}"
    target_column = "target"
    smiles_column = "smiles"

    def _infer_seed_from_split(split_value: str) -> int | None:
        if not split_value:
            return None
        text = str(split_value)
        if text.startswith("split"):
            suffix = text[5:]
            if suffix.isdigit():
                return int(suffix)
        return None

    try:
        lo, hi = resolve_thresholds(
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            thresholds_json=thresholds_json,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            seed=_infer_seed_from_split(split_column),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    path = write_threshold_classifier_dataset(
        input_path=input_path,
        output_path=output_path,
        smiles_column=smiles_column,
        split_column=split_column,
        metric_column=metric_column,
        target_column=target_column,
        lower_threshold=lo,
        upper_threshold=hi,
        compound_min_screens=compound_min_screens,
        compound_screens_column=compound_screens_column,
        screens_weight_mode=screens_weight_mode.value,
    )
    LOGGER.info("Wrote %s", path)
    typer.echo(path)


@app.command("trim-columns")
def trim_columns(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output Parquet file."),
    drop_columns: list[str] = typer.Option(
        [],
        "--drop-column",
        help="Column to drop (repeatable).",
    ),
    keep_columns: list[str] = typer.Option(
        [],
        "--keep-column",
        help="Column to retain even if other columns are dropped (repeatable).",
    ),
    keep_numeric_columns: bool = typer.Option(
        False,
        "--keep-numeric-columns",
        help="Retain columns whose names consist purely of digits.",
    ),
) -> None:
    """Drop selected columns to reduce dataset size."""
    path = write_trimmed_dataset(
        input_path=input_path,
        output_path=output_path,
        drop_columns=drop_columns or [],
        keep_numeric_columns=keep_numeric_columns,
        keep_columns=(keep_columns or None),
    )
    LOGGER.info("Wrote %s", path)
    typer.echo(path)


@app.command("submit-jobs")
def submit_jobs(
    config_path: Path = typer.Option(..., "--config", help="Path to the submission YAML file."),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for generated shell scripts."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Generate scripts without invoking qsub."
    ),
    submit_override: bool | None = typer.Option(
        None,
        "--submit/--no-submit",
        help="Force submit or skip execution regardless of config.",
    ),
) -> None:
    """Generate UGE-ready Chemprop scripts and optionally submit them via qsub."""
    try:
        global_cfg, jobs = load_submission_plan(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.models_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loaded %d job(s) from %s", len(jobs), config_path)

    def _submit_with_qsub(script_path: Path) -> None:
        cmd = ["qsub", str(script_path)]
        LOGGER.info("Submitting %s via %s", script_path, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise typer.Exit(1) from exc
        except subprocess.CalledProcessError as exc:
            raise typer.Exit(exc.returncode) from exc

    _write_scripts_and_maybe_submit(
        jobs=jobs,
        output_dir=output_dir,
        submit_override=submit_override,
        dry_run=dry_run,
        submitter=_submit_with_qsub,
        submit_label="qsub submission",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

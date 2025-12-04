from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence, cast

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TaskID,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .datasets import (
    resolve_thresholds,
    write_regression_dataset,
    write_smiles_csv,
    write_threshold_classifier_dataset,
    write_trimmed_dataset,
)
from .jobgen import ConfigError, JobDefinition, load_submission_plan

LOGGER = logging.getLogger("model_jobs.cli")
app = typer.Typer(
    no_args_is_help=True,
    help="Utilities for preparing Chemprop training and prediction inputs.",
)

PROGRESS_COLUMNS = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
)


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
    progress: Progress | None = None
    task_id: TaskID | None = None
    if len(jobs) > 1:
        progress = Progress(*PROGRESS_COLUMNS)
        progress.start()
        task_id = progress.add_task(f"{submit_label.title()} in progress", total=len(jobs))

    try:
        for job in jobs:
            if progress and task_id is not None:
                progress.update(task_id, description=f"{submit_label.title()}: {job.job_name}")

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

            if progress and task_id is not None:
                progress.advance(task_id)

        if progress and task_id is not None:
            progress.update(task_id, description=f"[green]{submit_label.title()} complete")
    finally:
        if progress:
            progress.stop()


def _parse_filter_value(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging.")) -> None:
    _setup_logging(verbose)


@app.command("prepare-regression")
def prepare_regression(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output Parquet file."),
    smiles_column: str = typer.Option("smiles", "--smiles-column", help="Column containing SMILES strings."),
    split_column: str = typer.Option("split", "--split-column", help="Column containing split assignments."),
    target_column: str = typer.Option("score", "--target-column", help="Regression target column."),
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
) -> None:
    """Build a regression-ready Parquet file with SMILES, split, and target columns."""
    path = write_regression_dataset(
        input_path=input_path,
        output_path=output_path,
        smiles_column=smiles_column,
        split_column=split_column,
        target_column=target_column,
        compound_min_screens=compound_min_screens,
        compound_screens_column=compound_screens_column,
    )
    LOGGER.info("Wrote %s", path)
    typer.echo(path)


@app.command("prepare-threshold-classifier")
def prepare_threshold_classifier(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output Parquet file."),
    smiles_column: str = typer.Option("smiles", "--smiles-column", help="Column containing SMILES strings."),
    split_column: str = typer.Option("split", "--split-column", help="Column containing split assignments."),
    target_column: str = typer.Option("target", "--target-column", help="Column containing the discrete label."),
    metric_column: str = typer.Option("score", "--metric-column", help="Continuous column to derive labels."),
    lower_threshold: float | None = typer.Option(None, "--lower-threshold", help="Absolute lower threshold."),
    upper_threshold: float | None = typer.Option(None, "--upper-threshold", help="Absolute upper threshold."),
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
) -> None:
    """Filter a dataset to binary labels and emit a classification Parquet file."""
    try:
        lo, hi = resolve_thresholds(
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            thresholds_json=thresholds_json,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
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
    )
    LOGGER.info("Wrote %s", path)
    typer.echo(path)


@app.command("prepare-smiles-csv")
def prepare_smiles_csv(
    input_path: Path = typer.Option(..., "--input", help="Source Parquet file."),
    output_path: Path = typer.Option(..., "--output", help="Path to the output CSV file."),
    smiles_column: str = typer.Option("smiles", "--smiles-column", help="Column containing SMILES strings."),
    filters: list[str] = typer.Option(
        [],
        "--filter",
        help="Filter rows where COLUMN equals VALUE (repeatable; COLUMN=VALUE).",
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
) -> None:
    """Extract SMILES (optionally filtered) to a CSV file for Chemprop prediction."""
    parsed_filters: dict[str, list[object]] = {}
    for raw in filters:
        if "=" not in raw:
            raise typer.BadParameter(f"Invalid filter '{raw}'. Expected COLUMN=VALUE.")
        column, value = raw.split("=", 1)
        column = column.strip()
        value = value.strip()
        if not column:
            raise typer.BadParameter(f"Invalid filter '{raw}': column name is empty.")
        parsed_filters.setdefault(column, []).append(_parse_filter_value(value))

    path = write_smiles_csv(
        input_path=input_path,
        output_path=output_path,
        smiles_column=smiles_column,
        filters=cast(dict[str, Sequence[object]] | None, parsed_filters or None),
        compound_min_screens=compound_min_screens,
        compound_screens_column=compound_screens_column,
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
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for generated shell scripts."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate scripts without executing them."),
    submit_override: bool | None = typer.Option(
        None,
        "--submit/--no-submit",
        help="Force submit or skip execution regardless of config.",
    ),
) -> None:
    """Generate Chemprop training/prediction scripts from YAML and optionally execute them."""
    try:
        global_cfg, jobs = load_submission_plan(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.models_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loaded %d job(s) from %s", len(jobs), config_path)

    def _execute_with_bash(script_path: Path) -> None:
        LOGGER.info("Executing %s via bash", script_path)
        try:
            subprocess.run(["bash", str(script_path)], check=True)
        except subprocess.CalledProcessError as exc:
            raise typer.Exit(exc.returncode) from exc

    _write_scripts_and_maybe_submit(
        jobs=jobs,
        output_dir=output_dir,
        submit_override=submit_override,
        dry_run=dry_run,
        submitter=_execute_with_bash,
        submit_label="execution",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

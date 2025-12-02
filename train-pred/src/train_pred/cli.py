from __future__ import annotations

import argparse
import ast
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .datasets import (
    resolve_thresholds,
    write_regression_dataset,
    write_smiles_csv,
    write_threshold_classifier_dataset,
    write_trimmed_dataset,
)
from .jobgen import ConfigError, load_submission_plan

LOGGER = logging.getLogger("train_pred.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _add_common_columns_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_split: bool = True,
) -> None:
    parser.add_argument("--smiles-column", default="smiles", help="Column containing SMILES strings.")
    if include_split:
        parser.add_argument("--split-column", default="split", help="Column containing split assignments.")


def _add_compound_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compound-min-screens",
        type=float,
        help="Filter out compounds with fewer screens than this threshold.",
    )
    parser.add_argument(
        "--compound-screens-column",
        default="screens",
        help="Column holding the per-compound screen count from the processed pipeline outputs.",
    )


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-pred",
        description="Utilities for preparing Chemprop training and prediction inputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # Regression
    reg_parser = subparsers.add_parser(
        "prepare-regression",
        help="Build a regression-ready Parquet file with SMILES, split, and target columns.",
    )
    reg_parser.add_argument("--input", required=True, help="Source Parquet file.")
    reg_parser.add_argument("--output", required=True, help="Path to the output Parquet file.")
    _add_common_columns_arguments(reg_parser)
    reg_parser.add_argument("--target-column", default="score", help="Regression target column.")
    _add_compound_filter_arguments(reg_parser)
    reg_parser.set_defaults(handler=_handle_prepare_regression)

    # Threshold classifier
    thr_parser = subparsers.add_parser(
        "prepare-threshold-classifier",
        help="Filter a dataset to binary labels and emit a classification Parquet file.",
    )
    thr_parser.add_argument("--input", required=True, help="Source Parquet file.")
    thr_parser.add_argument("--output", required=True, help="Path to the output Parquet file.")
    _add_common_columns_arguments(thr_parser)
    thr_parser.add_argument("--target-column", default="target", help="Column containing the discrete label.")
    _add_compound_filter_arguments(thr_parser)
    thr_parser.add_argument(
        "--metric-column",
        default="score",
        help="Continuous column used to derive binary labels.",
    )
    thr_parser.add_argument(
        "--lower-threshold",
        type=float,
        help="Absolute lower threshold; keep values <= threshold as class 0.",
    )
    thr_parser.add_argument(
        "--upper-threshold",
        type=float,
        help="Absolute upper threshold; keep values >= threshold as class 1.",
    )
    thr_parser.add_argument(
        "--thresholds-json",
        help="Statistics metadata (JSON) mapping percentiles to threshold values.",
    )
    thr_parser.add_argument(
        "--lower-percentile",
        type=float,
        help="Percentile to use as the lower threshold (requires --thresholds-json).",
    )
    thr_parser.add_argument(
        "--upper-percentile",
        type=float,
        help="Percentile to use as the upper threshold (requires --thresholds-json).",
    )
    thr_parser.set_defaults(handler=_handle_prepare_threshold_classifier)

    # SMILES CSV for prediction
    pred_parser = subparsers.add_parser(
        "prepare-smiles-csv",
        help="Extract SMILES (optionally filtered) to a CSV file for Chemprop prediction.",
    )
    pred_parser.add_argument("--input", required=True, help="Source Parquet file.")
    pred_parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    pred_parser.add_argument("--smiles-column", default="smiles", help="Column containing SMILES strings.")
    _add_compound_filter_arguments(pred_parser)
    pred_parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=[],
        metavar="COLUMN=VALUE",
        help="Filter rows where COLUMN equals VALUE (repeatable; multiple values per column allowed).",
    )
    pred_parser.set_defaults(handler=_handle_prepare_smiles_csv)

    # Trim columns
    trim_parser = subparsers.add_parser(
        "trim-columns",
        help="Drop selected columns to reduce dataset size.",
    )
    trim_parser.add_argument("--input", required=True, help="Source Parquet file.")
    trim_parser.add_argument("--output", required=True, help="Path to the output Parquet file.")
    trim_parser.add_argument(
        "--drop-column",
        action="append",
        dest="drop_columns",
        default=[],
        help="Column to drop (repeatable).",
    )
    trim_parser.add_argument(
        "--keep-column",
        action="append",
        dest="keep_columns",
        default=[],
        help="Column to retain even if other columns are dropped (repeatable).",
    )
    trim_parser.add_argument(
        "--keep-numeric-columns",
        action="store_true",
        help="Retain columns whose names consist purely of digits.",
    )
    trim_parser.set_defaults(handler=_handle_trim_columns)

    submit_parser = subparsers.add_parser(
        "submit-jobs",
        help="Generate Chemprop training/prediction scripts from a YAML config and optionally execute them.",
    )
    submit_parser.add_argument("--config", required=True, help="Path to the submission YAML file.")
    submit_parser.add_argument("--output-dir", required=True, help="Directory for generated shell scripts.")
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without executing them.",
    )
    submit_group = submit_parser.add_mutually_exclusive_group()
    submit_group.add_argument(
        "--submit",
        action="store_const",
        const=True,
        dest="submit_override",
        help="Submit all generated scripts regardless of config settings.",
    )
    submit_group.add_argument(
        "--no-submit",
        action="store_const",
        const=False,
        dest="submit_override",
        help="Do not submit scripts (overrides config).",
    )
    submit_parser.set_defaults(submit_override=None)
    submit_parser.set_defaults(handler=_handle_submit_jobs)

    return parser


def _handle_prepare_regression(args: argparse.Namespace) -> Path:
    return write_regression_dataset(
        input_path=args.input,
        output_path=args.output,
        smiles_column=args.smiles_column,
        split_column=args.split_column,
        target_column=args.target_column,
        compound_min_screens=args.compound_min_screens,
        compound_screens_column=args.compound_screens_column,
    )


def _handle_prepare_threshold_classifier(args: argparse.Namespace) -> Path:
    lower_threshold, upper_threshold = resolve_thresholds(
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
        thresholds_json=args.thresholds_json,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    return write_threshold_classifier_dataset(
        input_path=args.input,
        output_path=args.output,
        smiles_column=args.smiles_column,
        split_column=args.split_column,
        metric_column=args.metric_column,
        target_column=args.target_column,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        compound_min_screens=args.compound_min_screens,
        compound_screens_column=args.compound_screens_column,
    )


def _handle_prepare_smiles_csv(args: argparse.Namespace) -> Path:
    filters: dict[str, list[object]] = {}
    for raw in args.filters:
        if "=" not in raw:
            raise ValueError(f"Invalid filter '{raw}'. Expected COLUMN=VALUE.")
        column, value = raw.split("=", 1)
        column = column.strip()
        value = value.strip()
        if not column:
            raise ValueError(f"Invalid filter '{raw}': column name is empty.")
        filters.setdefault(column, []).append(_parse_filter_value(value))

    return write_smiles_csv(
        input_path=args.input,
        output_path=args.output,
        smiles_column=args.smiles_column,
        filters=filters or None,
        compound_min_screens=args.compound_min_screens,
        compound_screens_column=args.compound_screens_column,
    )


def _handle_trim_columns(args: argparse.Namespace) -> Path:
    return write_trimmed_dataset(
        input_path=args.input,
        output_path=args.output,
        drop_columns=args.drop_columns or [],
        keep_numeric_columns=args.keep_numeric_columns,
        keep_columns=(args.keep_columns or None),
    )


def _handle_submit_jobs(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    try:
        global_cfg, jobs = load_submission_plan(config_path)
    except ConfigError as exc:
        LOGGER.error("Invalid submission configuration: %s", exc)
        raise SystemExit(1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.models_dir.mkdir(parents=True, exist_ok=True)
    global_cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    override = args.submit_override
    dry_run = bool(args.dry_run)

    LOGGER.info("Loaded %d job(s) from %s", len(jobs), config_path)

    for job in jobs:
        script_path = output_dir / job.filename
        content = job.content if job.content.endswith("\n") else f"{job.content}\n"
        script_path.write_text(content)
        os.chmod(script_path, 0o750)

        should_submit = job.submit if override is None else override

        if dry_run:
            LOGGER.info("[DRY-RUN] Generated %s (execution skipped)", script_path)
            continue

        if should_submit:
            LOGGER.info("Executing %s via bash", script_path)
            try:
                subprocess.run(["bash", str(script_path)], check=True)
            except subprocess.CalledProcessError as exc:
                LOGGER.error("Execution failed for %s with exit code %s", script_path, exc.returncode)
                raise SystemExit(exc.returncode) from exc
        else:
            LOGGER.info("Generated %s (execution skipped by config)", script_path)

    return 0


def _parse_filter_value(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    handler: Callable[[argparse.Namespace], object] = args.handler
    result = handler(args)
    if isinstance(result, Path):
        LOGGER.info("Wrote %s", result)
        return 0
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

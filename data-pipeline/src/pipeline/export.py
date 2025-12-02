"""Dataset export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

EVAL_PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


def _boolean_cast_expressions(columns: Iterable[str]) -> list[pl.Expr]:
    return [
        pl.col(col).cast(pl.Boolean).alias(col) for col in columns
    ]


def _pivot_multilabel_matrix(
    retained_data_lf: pl.LazyFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Pivot assay activity into a wide multi-label matrix keyed by SMILES."""
    pivot_ready = (
        retained_data_lf.select(["smiles", "assay_id", pl.col("active").alias("active")])
        .group_by(["smiles", "assay_id"])
        .agg(pl.col("active").max())
        .rename({"active": "active_value"})
        .collect(engine="streaming")
    )

    matrix_df = (
        pivot_ready.pivot(
            values="active_value",
            index="smiles",
            on="assay_id",
            aggregate_function="first",
        )
        .sort("smiles")
    )

    assay_cols = [col for col in matrix_df.columns if col != "smiles"]
    if assay_cols:
        matrix_df = matrix_df.with_columns(_boolean_cast_expressions(assay_cols))
    return matrix_df, assay_cols


def _compute_thresholds(values: np.ndarray) -> dict[str, float]:
    """Return percentile → threshold mapping for the supplied hit-rate array."""
    if values.size == 0:
        raise ValueError("Cannot compute thresholds from an empty reliability subset.")
    thresholds = {
        str(percentile): float(np.nanpercentile(values, percentile))
        for percentile in EVAL_PERCENTILES
    }
    return thresholds


def write_model_datasets(
    assay_format: str,
    retained_data_lf: pl.LazyFrame,
    all_compounds_df: pl.DataFrame,
    split_map_df: pl.DataFrame,
    output_dir: Path,
    *,
    filter_threshold: float,
) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    """
    Write train/validation, calibration, and test datasets plus threshold metadata.

    Returns:
        dataset_counts: row counts for each emitted Parquet dataset
        threshold_percentiles: mapping from metric name to percentile → threshold values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    compounds_with_split = all_compounds_df.join(
        split_map_df.select(["smiles", "split"]), on="smiles", how="left"
    )

    multilabel_matrix, assay_cols = _pivot_multilabel_matrix(retained_data_lf)
    compound_features = compounds_with_split.join(multilabel_matrix, on="smiles", how="left")
    if assay_cols:
        compound_features = compound_features.with_columns(_boolean_cast_expressions(assay_cols))
    if {"hits", "screens"} <= set(compound_features.columns):
        compound_features = compound_features.with_columns(
            [
                pl.col("hits").fill_null(0),
                pl.col("screens").fill_null(0),
            ]
        )

    # Common masks
    reliability_mask = pl.col("passes_reliability_filter")
    split_series = pl.col("split")
    trainval_mask = split_series.is_in(["train", "val"])

    regression_cols = ["smiles", "split", "passes_reliability_filter"]
    if "compound_id" in compound_features.columns:
        regression_cols.insert(1, "compound_id")
    regression_cols.extend(
        [
            "hits",
            "screens",
            "hit_rate",
            "score",
        ]
    )

    regression_trainval_df = (
        compound_features.filter(trainval_mask)
        .select(regression_cols)
        .sort(["split", "smiles"])
    )
    regression_path = output_dir / f"{assay_format}_regression_trainval.parquet"
    regression_trainval_df.write_parquet(regression_path)

    multilabel_cols = ["smiles", "split", "passes_reliability_filter"]
    if "compound_id" in compound_features.columns:
        multilabel_cols.insert(1, "compound_id")
    multilabel_cols.extend(
        [
            "hits",
            "screens",
            "hit_rate",
            "score",
        ]
    )
    multilabel_cols.extend(col for col in assay_cols if col in compound_features.columns)

    multilabel_trainval_df = (
        compound_features.filter(trainval_mask & (pl.col("screens") > 0))
        .select(multilabel_cols)
        .sort(["split", "smiles"])
    )
    multilabel_path = output_dir / f"{assay_format}_multilabel_trainval.parquet"
    multilabel_trainval_df.write_parquet(multilabel_path)

    calibration_df = (
        compound_features.filter(split_series == "calibration")
        .drop(["split"])
        .sort("smiles")
    )
    calibration_path = output_dir / f"{assay_format}_calibration.parquet"
    calibration_df.write_parquet(calibration_path)

    test_df = (
        compound_features.filter(split_series == "test")
        .drop(["split"])
        .sort("smiles")
    )
    test_path = output_dir / f"{assay_format}_test.parquet"
    test_df.write_parquet(test_path)

    reliable_values = (
        compound_features.filter(reliability_mask)["score"]
        .drop_nulls()
        .to_numpy()
    )
    threshold_percentiles = _compute_thresholds(reliable_values)

    metadata = {
        "assay_format": assay_format,
        "filter_mode": "min_screens",
        "filter_threshold": float(filter_threshold),
        "percentiles": {"score": threshold_percentiles},
        "compound_count": int(compound_features.filter(reliability_mask).height),
    }
    thresholds_path = output_dir / f"{assay_format}_thresholds.json"
    thresholds_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    counts = {
        "regression_trainval": regression_trainval_df.height,
        "multilabel_trainval": multilabel_trainval_df.height,
        "calibration": calibration_df.height,
        "test": test_df.height,
    }
    thresholds_payload: dict[str, dict[str, float]] = {"score": threshold_percentiles}
    return counts, thresholds_payload

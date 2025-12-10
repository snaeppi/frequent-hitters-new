"""Dataset export helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

EVAL_PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


def _boolean_cast_expressions(columns: Iterable[str]) -> list[pl.Expr]:
    return [pl.col(col).cast(pl.Boolean).alias(col) for col in columns]


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

    matrix_df = pivot_ready.pivot(
        values="active_value",
        index="smiles",
        on="assay_id",
        aggregate_function="first",
    ).sort("smiles")

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
    reg_split_df: pl.DataFrame,
    mt_split_df: pl.DataFrame,
    output_dir: Path,
    *,
    filter_threshold: float,
    seeds: list[int],
) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    """
    Write unified regression and multi-task datasets plus threshold metadata.

    Returns:
        dataset_counts: row counts for each emitted Parquet dataset
        threshold_percentiles: mapping from metric name to percentile → threshold values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not seeds:
        raise ValueError("At least one seed is required to write datasets.")

    reg_split_columns = [f"split_seed{seed}" for seed in seeds]
    mt_split_columns = [f"split_seed{seed}" for seed in seeds]
    missing_reg = [col for col in reg_split_columns if col not in reg_split_df.columns]
    missing_mt = [col for col in mt_split_columns if col not in mt_split_df.columns]
    if missing_reg or missing_mt:
        missing = ", ".join(sorted(set(missing_reg + missing_mt)))
        raise ValueError(f"Missing split columns: {missing}")

    multilabel_matrix, assay_cols = _pivot_multilabel_matrix(retained_data_lf)

    def _prepare_frame(
        df: pl.DataFrame,
        split_cols: list[str],
        *,
        include_assay_columns: bool = True,
    ) -> pl.DataFrame:
        prepared = df
        bool_cols = [
            col
            for col in ["regression_eligible"]
            if col in prepared.columns
        ]
        if bool_cols:
            prepared = prepared.with_columns(
                [pl.col(col).cast(pl.Boolean).fill_null(False).alias(col) for col in bool_cols]
            )
        if {"hits", "screens"} <= set(prepared.columns):
            prepared = prepared.with_columns(
                [
                    pl.col("hits").fill_null(0),
                    pl.col("screens").fill_null(0),
                ]
            )
        if include_assay_columns and assay_cols:
            prepared = prepared.join(multilabel_matrix, on="smiles", how="left")
            prepared = prepared.with_columns(_boolean_cast_expressions(assay_cols))
        base_cols: list[str] = ["smiles"]
        base_cols.extend(
            [
                col
                for col in ["regression_eligible"]
                if col in prepared.columns
            ]
        )
        base_cols.extend(
            [
                col
                for col in ["hits", "screens", "hit_rate", "scaffold_smiles"]
                if col in prepared.columns
            ]
        )
        score_cols = [
            col
            for col in sorted(prepared.columns)
            if col.startswith("score_seed") or col == "score"
        ]
        base_cols.extend(score_cols)
        assay_selection = (
            [col for col in assay_cols if col in prepared.columns] if include_assay_columns else []
        )
        return prepared.select(base_cols + split_cols + assay_selection)

    regression_df = _prepare_frame(
        reg_split_df.filter(pl.col("regression_eligible")),
        reg_split_columns,
        include_assay_columns=False,
    ).sort("smiles")

    multilabel_df = _prepare_frame(mt_split_df, mt_split_columns).sort("smiles")
    multilabel_path = output_dir / f"{assay_format}_multilabel.parquet"
    multilabel_df.write_parquet(multilabel_path)

    reliability_mask = pl.col("regression_eligible")

    percentiles_by_seed: dict[str, dict[str, float]] = {}
    score_column_by_seed: dict[str, str] = {}
    compound_counts_by_seed: dict[str, int] = {}

    for seed in seeds:
        split_col = f"split_seed{seed}"
        if split_col not in regression_df.columns:
            raise KeyError(f"Missing split column '{split_col}' in regression dataset.")

        candidate_score_col = f"score_seed{seed}"
        if candidate_score_col not in regression_df.columns:
            candidate_score_col = "score"
        if candidate_score_col not in regression_df.columns:
            raise KeyError(
                f"No score column found for seed {seed}. Expected '{candidate_score_col}'."
            )

        subset = regression_df.filter(reliability_mask & (pl.col(split_col) == "train"))
        values = subset[candidate_score_col].drop_nulls().to_numpy()
        percentiles_by_seed[str(seed)] = _compute_thresholds(values)
        score_column_by_seed[str(seed)] = candidate_score_col
        compound_counts_by_seed[str(seed)] = int(subset.height)

    # Drop columns that are irrelevant for the regression output.
    regression_df = regression_df.drop("regression_eligible", missing="ignore")

    regression_path = output_dir / f"{assay_format}_regression.parquet"
    regression_df.write_parquet(regression_path)

    metadata = {
        "assay_format": assay_format,
        "filter_mode": "min_screens",
        "filter_threshold": float(filter_threshold),
        "seeds": [int(seed) for seed in seeds],
        "percentiles_by_seed": percentiles_by_seed,
        "score_column_by_seed": score_column_by_seed,
        "compound_counts_by_seed": compound_counts_by_seed,
    }

    thresholds_path = output_dir / f"{assay_format}_thresholds.json"
    thresholds_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    counts = {
        "regression": regression_df.height,
        "multilabel": multilabel_df.height,
    }
    thresholds_payload: dict[str, dict[str, float]] = percentiles_by_seed
    return counts, thresholds_payload

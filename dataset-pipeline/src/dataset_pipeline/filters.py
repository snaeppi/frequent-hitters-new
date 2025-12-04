"""Assay-level filtering and diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from . import viz

PLOT_DIRNAME = "plots"


def apply_assay_filters(
    data_lf: pl.LazyFrame,
    output_dir: Path,
    *,
    std_k: float,
    min_screens_per_assay: int,
    enable_plots: bool,
) -> tuple[
    pl.LazyFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, float],
]:
    """
    Compute per-assay statistics, apply exclusion rules, and persist artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_screens_per_assay < 1:
        raise ValueError("min_screens_per_assay must be >= 1.")

    agg = (
        data_lf.group_by("assay_id")
        .agg(
            [
                pl.len().alias("n_screens"),
                pl.col("active").cast(pl.Int64).sum().alias("n_hits"),
            ]
        )
        .with_columns(
            [
                pl.col("n_screens").cast(pl.UInt32).alias("n_screens"),
                pl.col("n_hits").cast(pl.UInt32).alias("n_hits"),
            ]
        )
        .with_columns(
            pl.when(pl.col("n_screens") > 0)
            .then(pl.col("n_hits") / pl.col("n_screens"))
            .otherwise(0.0)
            .cast(pl.Float64)
            .alias("hit_rate")
        )
    )

    assay_metadata = agg.collect(engine="streaming")
    if assay_metadata.is_empty():
        raise ValueError("No assays available after aggregation.")

    assay_metadata = assay_metadata.with_columns(
        (pl.col("n_screens") >= min_screens_per_assay).alias("meets_min_screens")
    )

    eligible_assays = assay_metadata.filter(pl.col("meets_min_screens"))
    if eligible_assays.is_empty():
        raise ValueError(
            "No assays satisfy the minimum screen requirement. "
            f"Adjust 'min_screens_per_assay' (current={min_screens_per_assay})."
        )

    mean_hit_rate = float(eligible_assays["hit_rate"].mean())
    std_raw = eligible_assays["hit_rate"].std()
    std_hit_rate = float(std_raw) if std_raw is not None else 0.0
    if not math.isfinite(std_hit_rate):
        std_hit_rate = 0.0
    threshold = mean_hit_rate + std_k * std_hit_rate if std_hit_rate > 0 else float("inf")

    outlier_expr = (
        pl.when(pl.col("meets_min_screens"))
        .then(pl.col("hit_rate") >= pl.lit(threshold))
        .otherwise(pl.lit(False))
        if std_hit_rate > 0
        else pl.lit(False)
    )
    insufficient_expr = pl.col("meets_min_screens").not_()

    assay_metadata = assay_metadata.with_columns(
        [
            outlier_expr.alias("excluded_due_to_hit_rate"),
            insufficient_expr.alias("excluded_due_to_min_screens"),
        ]
    )

    assay_metadata = assay_metadata.with_columns(
        [
            (
                pl.col("excluded_due_to_min_screens") | pl.col("excluded_due_to_hit_rate")
            ).alias("excluded_flag"),
            pl.when(pl.col("excluded_due_to_min_screens"))
            .then(pl.lit("min_screens"))
            .when(pl.col("excluded_due_to_hit_rate") & outlier_expr)
            .then(pl.lit("hit_rate_outlier"))
            .otherwise(pl.lit(None))
            .alias("exclude_reason"),
        ]
    )

    assays_after_hit_rate = assay_metadata.filter(pl.col("excluded_due_to_hit_rate").not_())
    retained_assays = assays_after_hit_rate.filter(pl.col("excluded_due_to_min_screens").not_())
    excluded_assays = assay_metadata.filter(pl.col("excluded_flag"))

    assay_metadata.write_parquet(output_dir / "assay_metadata.parquet")
    retained_assays.write_parquet(output_dir / "retained_assays.parquet")
    excluded_assays.write_parquet(output_dir / "outlier_assays.parquet")

    retained_ids_df = retained_assays.select("assay_id")
    filtered_lf = data_lf.join(retained_ids_df.lazy(), on="assay_id", how="inner")

    stats = {
        "mean_hit_rate": mean_hit_rate,
        "std_hit_rate": std_hit_rate,
        "threshold_hit_rate": threshold,
        "eligible_assays": int(eligible_assays.height),
        "min_screens_per_assay": int(min_screens_per_assay),
        "assays_after_hit_rate_filter": int(assays_after_hit_rate.height),
        "assays_retained": int(retained_assays.height),
        "assays_excluded": int(excluded_assays.height),
    }

    if enable_plots:
        plot_dir = output_dir / PLOT_DIRNAME
        plot_dir.mkdir(exist_ok=True)
        viz.plot_assay_hit_rate_distribution(
            assay_metadata,
            mean_hit_rate,
            std_hit_rate,
            std_k,
            plot_dir / "assay_hit_rate_distribution.png",
            plot_dir / "assay_hit_rate_distribution.csv",
        )
        viz.plot_assay_coverage_hist(
            retained_assays,
            plot_dir / "assay_coverage_hist.png",
            plot_dir / "assay_coverage_hist.csv",
        )
        viz.plot_assay_positive_rate_hist(
            retained_assays,
            plot_dir / "assay_positive_rate_hist.png",
            plot_dir / "assay_positive_rate_hist.csv",
        )

    return filtered_lf, assay_metadata, retained_assays, excluded_assays, stats

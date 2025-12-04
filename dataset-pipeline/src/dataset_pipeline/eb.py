"""Empirical Bayes aggregation with a Method-of-Moments prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import viz


def _compute_prior(
    counts_df: pl.DataFrame,
    *,
    hits_col: str,
    screens_col: str,
    min_screens_for_prior_fit: int,
) -> tuple[float, float, float]:
    """
    Return (alpha0, beta0, mean_hit_rate) using a Method-of-Moments Beta prior.

    The prior is estimated from compound-level hit rates (hits/screens), matching
    the empirical mean and variance among compounds with at least
    `min_screens_for_prior_fit` measurements. Degenerate cases fall back to
    Beta(1, 1) with a mean of 0.5.
    """
    if hits_col not in counts_df.columns or screens_col not in counts_df.columns:
        raise KeyError(
            f"counts_df must contain '{hits_col}' and '{screens_col}' columns."
        )

    if counts_df.is_empty():
        return 1.0, 1.0, 0.5

    eligible = counts_df.filter(
        pl.col(screens_col) >= pl.lit(min_screens_for_prior_fit)
    )
    if eligible.is_empty():
        eligible = counts_df

    hits = eligible[hits_col].to_numpy().astype(np.float64, copy=False)
    screens = eligible[screens_col].to_numpy().astype(np.float64, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rates = np.divide(
            hits,
            screens,
            out=np.zeros_like(hits, dtype=np.float64),
            where=screens > 0,
        )

    hit_rates = hit_rates[np.isfinite(hit_rates)]
    n = int(hit_rates.size)
    if n == 0:
        return 1.0, 1.0, 0.5

    m = float(hit_rates.mean())
    if not np.isfinite(m):
        return 1.0, 1.0, 0.5

    if n < 2:
        return 1.0, 1.0, 0.5

    v = float(hit_rates.var(ddof=1))
    if not np.isfinite(v) or v <= 0.0 or m <= 0.0 or m >= 1.0:
        return 1.0, 1.0, 0.5

    common = (m * (1.0 - m)) / max(v, 1e-12) - 1.0
    a = max(common * m, 0.1)
    b = max(common * (1.0 - m), 0.1)
    return float(a), float(b), m


def _aggregate_compound_counts(
    data_lf: pl.LazyFrame,
    *,
    screens_col: str,
    hits_col: str,
) -> pl.DataFrame:
    """Aggregate per-compound screen and hit counts for the provided lazy frame."""
    group_keys = ["smiles"]
    schema_names = data_lf.collect_schema().names()
    include_compound_id = "compound_id" in schema_names

    value_exprs: list[pl.Expr] = [
        pl.len().alias(screens_col),
        pl.col("active").fill_null(0).cast(pl.UInt32).sum().alias(hits_col),
    ]
    if include_compound_id:
        value_exprs.append(
            pl.col("compound_id").drop_nulls().first().alias("compound_id")
        )

    aggregated = (
        data_lf.group_by(group_keys)
        .agg(value_exprs)
        .with_columns(
            [
                pl.col(screens_col).cast(pl.UInt32).alias(screens_col),
                pl.col(hits_col).cast(pl.UInt32).alias(hits_col),
            ]
        )
        .collect(engine="streaming")
    )
    return aggregated


def _attach_scores(
    df: pl.DataFrame,
    *,
    hits_col: str,
    screens_col: str,
    alpha0: float,
    beta0: float,
) -> pl.DataFrame:
    """Attach raw hit rates and EB scores to the provided DataFrame."""
    hits_np = df[hits_col].to_numpy().astype(np.float64, copy=False)
    screens_np = df[screens_col].to_numpy().astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rate_np = np.divide(
            hits_np,
            screens_np,
            out=np.zeros_like(hits_np, dtype=np.float64),
            where=screens_np > 0,
        )

    denom = screens_np + alpha0 + beta0
    with np.errstate(divide="ignore", invalid="ignore"):
        score_np = np.divide(
            hits_np + alpha0,
            denom,
            out=np.full_like(raw_rate_np, fill_value=np.nan, dtype=np.float64),
            where=denom > 0,
        )

    return df.with_columns(
        [
            pl.Series("hit_rate", raw_rate_np).cast(pl.Float64),
            pl.Series("score", score_np).cast(pl.Float64),
        ]
    )


def _retention_curve_from_screens(screens: np.ndarray) -> pl.DataFrame:
    """Compute retention fractions for unique minimum-screen thresholds."""
    unique_thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([1], dtype=np.int64),
                screens.astype(np.int64, copy=False),
            ]
        )
    )
    unique_thresholds.sort()
    total = float(len(screens))
    retention_data = [
        (int(threshold), float(np.sum(screens >= threshold) / total))
        for threshold in unique_thresholds
    ]
    return pl.DataFrame(
        {
            "min_screens": [val for val, _ in retention_data],
            "fraction_retained": [frac for _, frac in retention_data],
        }
    ).with_columns(
        [
            pl.col("min_screens").cast(pl.Int32),
            pl.col("fraction_retained").cast(pl.Float64),
        ]
    )


def build_compound_metadata(
    filtered_data_lf: pl.LazyFrame,
    output_dir: Path,
    *,
    min_screens_per_compound: int = 1,
    min_screens_for_prior_fit: int,
    enable_plots: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """
    Aggregate compound-level stats, apply Empirical Bayes shrinkage, and persist outputs.

    Returns a tuple of:
      - filtered compound DataFrame (only reliability-passing compounds)
      - full compound DataFrame annotated with EB score and a `passes_reliability_filter` flag
      - diagnostics dictionary for manifest logging
    """
    if min_screens_per_compound < 1:
        raise ValueError("min_screens_per_compound must be >= 1.")
    if min_screens_for_prior_fit < 1:
        raise ValueError("min_screens_for_prior_fit must be >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-compound counts over the selected assays
    compound_counts_df = _aggregate_compound_counts(
        filtered_data_lf,
        screens_col="screens",
        hits_col="hits",
    )

    total_compounds = compound_counts_df.height
    if total_compounds == 0:
        raise ValueError("No compounds found after upstream filtering.")

    alpha, beta, mean_hit_rate = _compute_prior(
        compound_counts_df,
        hits_col="hits",
        screens_col="screens",
        min_screens_for_prior_fit=min_screens_for_prior_fit,
    )

    scored_df = _attach_scores(
        compound_counts_df,
        hits_col="hits",
        screens_col="screens",
        alpha0=alpha,
        beta0=beta,
    )

    fit_expr = pl.col("screens") >= min_screens_for_prior_fit
    filter_expr = pl.col("screens") >= min_screens_per_compound
    filter_threshold: float = float(min_screens_per_compound)
    chosen_label = f"Chosen ≥ {int(min_screens_per_compound)}"

    all_compounds_flagged = scored_df.with_columns(
        [
            fit_expr.alias("meets_prior_fit_threshold"),
            filter_expr.alias("passes_reliability_filter"),
        ]
    )
    compound_df = all_compounds_flagged.filter(filter_expr)

    retained_compounds = compound_df.height
    if retained_compounds == 0:
        raise ValueError(
            "No compounds satisfy the filtering threshold. "
            "Adjust 'min_screens_per_compound' or upstream assay filtering."
        )

    # Persist compound metadata (full universe)
    all_compounds_flagged.write_parquet(output_dir / "compound_metadata.parquet")

    # Build diagnostics & plots from the full universe
    if enable_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        retention_df = _retention_curve_from_screens(
            all_compounds_flagged["screens"].to_numpy()
        )
        viz.plot_retention_curve(
            retention_df,
            filter_threshold,
            plot_dir / "retention_curve.png",
            plot_dir / "retention_curve.csv",
            threshold_column="min_screens",
            x_label="Minimum screens per compound",
            chosen_label=chosen_label if filter_threshold is not None else None,
        )

    diagnostics: dict[str, object] = {
        "total_compounds": total_compounds,
        "retained_compounds": retained_compounds,
        "passes_flagged": int(
            all_compounds_flagged.filter(pl.col("passes_reliability_filter")).height
        ),
        "filter_mode": "min_screens",
        "filter_threshold": filter_threshold,
        "prior_alpha": alpha,
        "prior_beta": beta,
        "prior_mode": "method_of_moments",
        "prior_mean_hit_rate": mean_hit_rate,
        "min_screens_for_prior_fit": min_screens_for_prior_fit,
    }
    return compound_df, all_compounds_flagged, diagnostics

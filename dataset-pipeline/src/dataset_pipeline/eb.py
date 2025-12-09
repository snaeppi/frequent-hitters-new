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
        raise KeyError(f"counts_df must contain '{hits_col}' and '{screens_col}' columns.")

    if counts_df.is_empty():
        return 1.0, 1.0, 0.5

    eligible = counts_df.filter(pl.col(screens_col) >= pl.lit(min_screens_for_prior_fit))
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
        value_exprs.append(pl.col("compound_id").drop_nulls().first().alias("compound_id"))

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


def _attach_score_column(
    df: pl.DataFrame,
    *,
    hits_col: str,
    screens_col: str,
    alpha0: float,
    beta0: float,
    score_col: str,
) -> pl.DataFrame:
    """Attach an EB score column computed with the provided prior."""
    hits_np = df[hits_col].to_numpy().astype(np.float64, copy=False)
    screens_np = df[screens_col].to_numpy().astype(np.float64, copy=False)

    denom = screens_np + alpha0 + beta0
    with np.errstate(divide="ignore", invalid="ignore"):
        score_np = np.divide(
            hits_np + alpha0,
            denom,
            out=np.full_like(screens_np, fill_value=np.nan, dtype=np.float64),
            where=denom > 0,
        )

    return df.with_columns([pl.Series(score_col, score_np).cast(pl.Float64)])


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


def compute_compound_counts(
    filtered_data_lf: pl.LazyFrame,
    *,
    min_screens_per_compound: int,
    min_screens_for_prior_fit: int,
    output_dir: Path,
    enable_plots: bool,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """
    Aggregate per-compound counts and attach reliability flags (no EB fitting).

    Returns the full compound DataFrame (including non-eligible compounds) and diagnostics.
    """
    if min_screens_per_compound < 1:
        raise ValueError("min_screens_per_compound must be >= 1.")
    if min_screens_for_prior_fit < 1:
        raise ValueError("min_screens_for_prior_fit must be >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    compound_counts_df = _aggregate_compound_counts(
        filtered_data_lf,
        screens_col="screens",
        hits_col="hits",
    )

    total_compounds = compound_counts_df.height
    if total_compounds == 0:
        raise ValueError("No compounds found after upstream filtering.")

    hits_np = compound_counts_df["hits"].to_numpy().astype(np.float64, copy=False)
    screens_np = compound_counts_df["screens"].to_numpy().astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rate_np = np.divide(
            hits_np,
            screens_np,
            out=np.zeros_like(hits_np, dtype=np.float64),
            where=screens_np > 0,
        )

    filter_expr = pl.col("screens") >= min_screens_per_compound
    prior_fit_expr = pl.col("screens") >= min_screens_for_prior_fit
    filter_threshold = float(min_screens_per_compound)
    chosen_label = f"Chosen ≥ {int(min_screens_per_compound)}"

    all_compounds = compound_counts_df.with_columns(
        [
            pl.Series("hit_rate", raw_rate_np).cast(pl.Float64),
            filter_expr.alias("passes_reliability_filter"),
            filter_expr.alias("regression_eligible"),
            prior_fit_expr.alias("meets_prior_fit_threshold"),
        ]
    )

    retained_compounds = int(all_compounds.filter(filter_expr).height)
    if retained_compounds == 0:
        raise ValueError(
            "No compounds satisfy the filtering threshold. "
            "Adjust 'min_screens_per_compound' or upstream assay filtering."
        )

    if enable_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        retention_df = _retention_curve_from_screens(all_compounds["screens"].to_numpy())
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
        "passes_flagged": int(all_compounds.filter(filter_expr).height),
        "filter_mode": "min_screens",
        "filter_threshold": filter_threshold,
        "min_screens_for_prior_fit": min_screens_for_prior_fit,
    }
    return all_compounds, diagnostics


def score_by_seed(
    *,
    regression_df: pl.DataFrame,
    multitask_df: pl.DataFrame,
    seeds: list[int],
    min_screens_for_prior_fit: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """
    Fit EB priors on the train split for each seed and attach per-seed score columns.

    Returns (regression_df_with_scores, multitask_df_with_scores, diagnostics).
    """
    if not seeds:
        raise ValueError("At least one seed is required for EB scoring.")
    if regression_df.is_empty():
        raise ValueError("Regression split is empty; cannot fit EB priors.")
    if min_screens_for_prior_fit < 1:
        raise ValueError("min_screens_for_prior_fit must be >= 1.")

    reg_scored = regression_df
    mt_scored = multitask_df
    priors: dict[str, object] = {}

    for seed in seeds:
        split_col = f"split_seed{seed}"
        score_col = f"score_seed{seed}"
        if split_col not in regression_df.columns:
            raise KeyError(f"Missing split column '{split_col}' for seed {seed}.")

        train_subset = regression_df.filter(pl.col(split_col) == "train")
        if train_subset.is_empty():
            raise ValueError(f"No training compounds found for seed {seed}.")

        alpha, beta, mean_hit_rate = _compute_prior(
            train_subset,
            hits_col="hits",
            screens_col="screens",
            min_screens_for_prior_fit=min_screens_for_prior_fit,
        )

        reg_scored = _attach_score_column(
            reg_scored,
            hits_col="hits",
            screens_col="screens",
            alpha0=alpha,
            beta0=beta,
            score_col=score_col,
        )
        mt_scored = _attach_score_column(
            mt_scored,
            hits_col="hits",
            screens_col="screens",
            alpha0=alpha,
            beta0=beta,
            score_col=score_col,
        )

        priors[str(seed)] = {
            "alpha": alpha,
            "beta": beta,
            "mean_hit_rate": mean_hit_rate,
            "train_compounds": int(train_subset.height),
            "train_compounds_meeting_fit_threshold": int(
                train_subset.filter(pl.col("screens") >= min_screens_for_prior_fit).height
            ),
        }

    if len(seeds) == 1:
        seed_col = f"score_seed{seeds[0]}"
        if seed_col in reg_scored.columns:
            reg_scored = reg_scored.with_columns(pl.col(seed_col).alias("score"))
            mt_scored = mt_scored.with_columns(pl.col(seed_col).alias("score"))

    diagnostics: dict[str, object] = {
        "prior_mode": "method_of_moments",
        "min_screens_for_prior_fit": min_screens_for_prior_fit,
        "priors_by_seed": priors,
    }
    return reg_scored, mt_scored, diagnostics

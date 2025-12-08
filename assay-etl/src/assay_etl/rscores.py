"""R-score computation and aggregation into a single Parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .polars_helpers import is_numeric_dtype, numeric_expr
from .selection import _replicate_base

CID_COLUMN = "PUBCHEM_CID"
SMILES_COLUMN_DEFAULT = "PUBCHEM_EXT_DATASOURCE_SMILES"


def _compute_median_mad(df: pl.DataFrame, column: str) -> tuple[float, float]:
    """Compute median and MAD for a column, treating convertible strings as numeric."""
    dtype = df.schema.get(column)
    expr = numeric_expr(column, dtype)
    median = df.select(expr.median().alias("median")).get_column("median").item()
    mad = df.select((expr - median).abs().median().alias("mad")).get_column("mad").item()
    return float(median), float(mad)


def _ensure_replicate_means(assay_parquet: Path) -> None:
    """Ensure replicate mean columns are materialized in the per-assay table.

    This matches the replicate-handling logic used during column selection so that
    selected columns like `_..._mean` exist when computing r-scores, even if
    stats were never computed for that assay in this run.
    """
    if not assay_parquet.exists():
        return

    lf = pl.scan_parquet(assay_parquet)
    schema = lf.collect_schema()

    # Find numeric columns that look like replicates.
    numeric_cols: list[str] = [name for name, dtype in schema.items() if is_numeric_dtype(dtype)]
    replicate_groups: dict[str, list[str]] = {}
    for name in numeric_cols:
        base = _replicate_base(name)
        if base:
            replicate_groups.setdefault(base, []).append(name)
    # As in selection._compute_candidate_stats, treat any detected replicate
    # group (including a single surviving member) as eligible for a mean
    # column so that `_..._mean` names are always available.
    replicate_means = replicate_groups
    if not replicate_means:
        return

    new_cols: list[pl.Expr] = []
    for base, cols in replicate_means.items():
        # Match the naming used in selection._compute_candidate_stats.
        new_name = f"_{base}_mean"
        if new_name in schema:
            continue
        value_exprs = [pl.col(c).cast(pl.Float64) for c in cols]
        mean_expr = pl.mean_horizontal(value_exprs).alias(new_name)
        new_cols.append(mean_expr)

    if not new_cols:
        return

    lf_with_means = lf.with_columns(new_cols)
    tmp_path = assay_parquet.with_suffix(".tmp.parquet")
    lf_with_means.sink_parquet(tmp_path, compression="zstd")
    tmp_path.replace(assay_parquet)


def compute_rscores_for_assay(
    *,
    aid: int,
    assay_parquet: Path,
    selected_column: str,
    median: float | None = None,
    mad: float | None = None,
    smiles_column: str = SMILES_COLUMN_DEFAULT,
) -> pl.DataFrame:
    """Compute r-scores for a single assay and return a small DataFrame."""
    if not assay_parquet.exists():
        raise FileNotFoundError(f"Assay table parquet not found: {assay_parquet}")

    # Ensure replicate mean columns (_..._mean) exist when needed.
    _ensure_replicate_means(assay_parquet)

    lf = pl.scan_parquet(assay_parquet)
    schema = lf.collect_schema()
    if CID_COLUMN not in schema:
        raise ValueError(f"Assay table parquet missing {CID_COLUMN}")
    if selected_column not in schema:
        raise ValueError(f"Selected column {selected_column} not found in assay table parquet.")

    if smiles_column not in schema:
        raise ValueError(f"Assay table parquet missing SMILES column {smiles_column}")

    if median is None or mad is None:
        df_small = lf.select(selected_column).collect()
        median, mad = _compute_median_mad(df_small, selected_column)

    if mad == 0:
        raise ValueError(f"MAD=0 for selected column {selected_column}, cannot compute r-scores.")

    base_expr = numeric_expr(selected_column, schema.get(selected_column))
    scaled_mad = mad * 1.4826
    r_expr = ((base_expr - pl.lit(median)) / pl.lit(scaled_mad)).alias("r_score")

    df = lf.select(
        pl.lit(aid).alias("assay_id"),
        pl.col(CID_COLUMN).alias("compound_id"),
        pl.col(smiles_column).alias("smiles"),
        r_expr,
    ).collect(engine="streaming")
    return df

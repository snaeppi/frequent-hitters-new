"""R-score computation and aggregation into a single Parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .meta_db import iter_selected_for_rscores
from .polars_helpers import is_numeric_dtype, numeric_expr
from .selection import _replicate_base

CID_COLUMN = "PUBCHEM_CID"
SMILES_COLUMN_DEFAULT = "PUBCHEM_EXT_DATASOURCE_SMILES"


def _compute_median_mad(df: pl.DataFrame, column: str) -> tuple[float, float]:
    """Compute median and MAD for a column, treating convertible strings as numeric."""
    dtype = df.schema.get(column)
    expr = numeric_expr(column, dtype)
    median = (
        df.select(expr.median().alias("median"))
        .get_column("median")
        .item()
    )
    mad = (
        df.select((expr - median).abs().median().alias("mad"))
        .get_column("mad")
        .item()
    )
    return float(median), float(mad)


def _ensure_replicate_means(aggregated_parquet: Path) -> None:
    """Ensure replicate mean columns are materialized in the aggregated file.

    This matches the replicate-handling logic used during column selection so that
    selected columns like `_..._mean` exist when computing r-scores, even if
    stats were never computed for that assay in this run.
    """
    if not aggregated_parquet.exists():
        return

    lf = pl.scan_parquet(aggregated_parquet)
    schema = lf.collect_schema()

    # Find numeric columns that look like replicates.
    numeric_cols: list[str] = [
        name for name, dtype in schema.items() if is_numeric_dtype(dtype)
    ]
    replicate_groups: dict[str, list[str]] = {}
    for name in numeric_cols:
        base = _replicate_base(name)
        if base:
            replicate_groups.setdefault(base, []).append(name)
    replicate_means = {
        base: cols for base, cols in replicate_groups.items() if len(cols) >= 2
    }
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
    tmp_path = aggregated_parquet.with_suffix(".tmp.parquet")
    lf_with_means.sink_parquet(tmp_path, compression="zstd")
    tmp_path.replace(aggregated_parquet)


def compute_rscores_for_assay(
    *,
    aid: int,
    aggregated_parquet: Path,
    selected_column: str,
    median: float | None = None,
    mad: float | None = None,
    smiles_column: str = SMILES_COLUMN_DEFAULT,
) -> pl.DataFrame:
    """Compute r-scores for a single assay and return a small DataFrame."""
    if not aggregated_parquet.exists():
        raise FileNotFoundError(f"Aggregated parquet not found: {aggregated_parquet}")

    # Ensure replicate mean columns (_..._mean) exist when needed.
    _ensure_replicate_means(aggregated_parquet)

    lf = pl.scan_parquet(aggregated_parquet)
    schema = lf.collect_schema()
    if CID_COLUMN not in schema:
        raise ValueError(f"Aggregated parquet missing {CID_COLUMN}")
    if selected_column not in schema:
        raise ValueError(
            f"Selected column {selected_column} not found in aggregated parquet."
        )

    if smiles_column not in schema:
        raise ValueError(f"Aggregated parquet missing SMILES column {smiles_column}")

    if median is None or mad is None:
        df_small = lf.select(selected_column).collect()
        median, mad = _compute_median_mad(df_small, selected_column)

    if mad == 0:
        raise ValueError(
            f"MAD=0 for selected column {selected_column}, cannot compute r-scores."
        )

    base_expr = numeric_expr(selected_column, schema.get(selected_column))
    scaled_mad = mad * 1.4826
    r_expr = (
        (base_expr - pl.lit(median)) / pl.lit(scaled_mad)
    ).alias("r_score")

    df = (
        lf.select(
            pl.lit(aid).alias("assay_id"),
            pl.col(CID_COLUMN).alias("compound_id"),
            pl.col(smiles_column).alias("smiles"),
            r_expr,
        )
        .collect(engine="streaming")
    )
    return df


def compute_rscores_from_metadata(
    *,
    metadata_db: Path,
    aggregated_dir: Path,
    output_parquet: Path,
    smiles_column: str = SMILES_COLUMN_DEFAULT,
) -> None:
    """Compute r-scores for all assays with a selected column."""
    outputs: list[pl.DataFrame] = []
    for aid, selected_column, median, mad in iter_selected_for_rscores(metadata_db):
        agg_path = aggregated_dir / f"aid_{aid}_cid_agg.parquet"
        df = compute_rscores_for_assay(
            aid=aid,
            aggregated_parquet=agg_path,
            selected_column=selected_column,
            median=median,
            mad=mad,
            smiles_column=smiles_column,
        )
        outputs.append(df)

    if outputs:
        result = pl.concat(outputs, how="diagonal")
    else:
        result = pl.DataFrame(
            {
                "assay_id": pl.Series([], dtype=pl.Int64),
                "compound_id": pl.Series([], dtype=pl.Int64),
                "smiles": pl.Series([], dtype=pl.String),
                "r_score": pl.Series([], dtype=pl.Float64),
            }
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_parquet)

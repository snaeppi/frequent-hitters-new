"""Compound-level aggregation to ensure one row per (AID, CID).

Assumes numeric columns are already typed in the source parquet (handled during download).
Implemented fully with lazy/streaming plans to avoid materializing the raw assay table.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

from .polars_helpers import is_numeric_dtype, numeric_expr

GROUP_KEY = "PUBCHEM_CID"
EXCLUDE_GROUP_COLUMNS = {"PUBCHEM_SID"}


def _categorize_columns(schema: dict[str, pl.DataType]) -> tuple[list[str], list[str]]:
    """Categorize columns into numeric and categorical based on schema dtypes."""
    numeric: list[str] = []
    categorical: list[str] = []

    for name, dtype in schema.items():
        if name == GROUP_KEY or name in EXCLUDE_GROUP_COLUMNS:
            continue

        if is_numeric_dtype(dtype):
            numeric.append(name)
        else:
            categorical.append(name)

    return numeric, categorical


def aggregate_compounds(
    *,
    aid: int,
    input_parquet: Path,
    output_parquet: Path,
) -> None:
    """Aggregate assay tables to one row per PUBCHEM_CID."""
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(input_parquet)
    schema = lf.collect_schema()

    if GROUP_KEY not in schema:
        raise ValueError(f"Input parquet {input_parquet} missing {GROUP_KEY}")

    numeric_cols, categorical_cols = _categorize_columns(schema)

    aggregations: list[pl.Expr] = []
    aggregations.extend(
        numeric_expr(col, schema[col]).median().alias(col) for col in numeric_cols
    )

    mode_aliases = {col: f"__mode_{col}" for col in categorical_cols}
    aggregations.extend(pl.col(col).mode().alias(alias) for col, alias in mode_aliases.items())

    if aggregations:
        aggregated = lf.group_by(GROUP_KEY).agg(aggregations)
    else:
        aggregated = lf.select(GROUP_KEY).unique()

    if mode_aliases:
        aggregated = (
            aggregated.with_columns(
                [pl.col(alias).list.first().alias(col) for col, alias in mode_aliases.items()]
            ).drop(list(mode_aliases.values()))
        )

    aggregated.sink_parquet(output_parquet)

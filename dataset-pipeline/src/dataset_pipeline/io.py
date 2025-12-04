"""Input/output helpers for the pipeline."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema


def _strip_empty_strings(column: pl.Expr) -> pl.Expr:
    """Convert empty strings to nulls to avoid sentinel values."""
    trimmed = column.str.strip_chars()
    return pl.when(trimmed == "").then(pl.lit(None)).otherwise(trimmed)


def _canonicalize_strings(cols: list[str]) -> list[pl.Expr]:
    """Cast and clean string columns."""
    return [
        _strip_empty_strings(pl.col(col).cast(pl.Utf8, strict=False)).alias(col)
        for col in cols
    ]


def load_assay_table(
    path: Path,
    expected_format: str,
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """
    Load an assay table from Parquet, enforce schema, and drop duplicates.

    Returns a lazy frame and a dictionary of row counts for QC logging.
    """
    if expected_format not in schema.VALID_ASSAY_FORMATS:
        valid = ", ".join(sorted(schema.VALID_ASSAY_FORMATS))
        raise ValueError(f"Unexpected assay_format '{expected_format}'. Valid: {valid}")

    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")

    lf = pl.scan_parquet(str(path))
    lf_columns = lf.collect_schema().names()
    schema.ensure_required_columns(lf_columns)

    optional_present = [col for col in schema.OPTIONAL_COLUMNS if col in lf_columns]
    ordered_cols = schema.canonical_column_order(optional_present)
    lf = lf.select(ordered_cols)

    string_cols = ["smiles", "assay_id"] + optional_present
    lf = lf.with_columns(_canonicalize_strings(string_cols))
    lf = lf.with_columns(
        pl.col("active").cast(pl.Int8, strict=False).alias("active"),
    )

    total_rows = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])

    lf = lf.with_row_index(name="_row_idx")
    value_columns = [col for col in ordered_cols if col not in {"smiles", "assay_id"}]
    dedup_lf = (
        lf.sort("_row_idx")
        .group_by(["smiles", "assay_id"], maintain_order=True)
        .agg(
            [pl.col("_row_idx").last().alias("_row_idx")]
            + [pl.col(col).last().alias(col) for col in value_columns]
        )
        .sort("_row_idx")
        .drop("_row_idx")
    )
    dedup_rows = int(dedup_lf.select(pl.len()).collect(engine="streaming")[0, 0])

    qc = {
        "rows_total": total_rows,
        "rows_after_dedup": dedup_rows,
    }
    return dedup_lf, qc

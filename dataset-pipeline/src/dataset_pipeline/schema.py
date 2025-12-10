"""Canonical schema definitions for the pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

REQUIRED_COLUMNS = {
    "smiles": pl.Utf8,
    "assay_id": pl.Utf8,
    "active": pl.Int8,
}

VALID_ASSAY_FORMATS = {"biochemical", "cellular"}


def ensure_required_columns(columns: Iterable[str]) -> None:
    """Validate that all required columns are present."""
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {joined}")


def canonical_column_order(_present_optional: Iterable[str] | None = None) -> list[str]:
    """Return a stable column order for downstream writes."""
    return list(REQUIRED_COLUMNS)

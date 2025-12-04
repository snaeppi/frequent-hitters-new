"""Shared Polars helpers."""

from __future__ import annotations

import polars as pl

_BASE_NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Decimal,
}
if hasattr(pl, "Int128"):
    _BASE_NUMERIC_DTYPES.add(getattr(pl, "Int128"))
if hasattr(pl, "UInt128"):
    _BASE_NUMERIC_DTYPES.add(getattr(pl, "UInt128"))

NUMERIC_DTYPES = frozenset(_BASE_NUMERIC_DTYPES)


def is_numeric_dtype(dtype: pl.DataType | None) -> bool:
    return dtype in NUMERIC_DTYPES


def numeric_expr(column: str, dtype: pl.DataType | None) -> pl.Expr:
    expr = pl.col(column)
    if not is_numeric_dtype(dtype):
        expr = expr.cast(pl.Float64, strict=False)
    return expr


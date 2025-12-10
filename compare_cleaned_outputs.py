#!/usr/bin/env python3
"""
Quick one-off script to compare old vs. new datasets by (assay_id, smiles).

Usage:
  python compare_cleaned_outputs.py /path/to/old.parquet /path/to/new.parquet [diff_out.csv]

You can restrict comparisons to specific columns with:
  --column hits --column screens
or by listing columns (one per line) in a text file:
  --columns-file cols.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _read_table(path: Path) -> pl.DataFrame:
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if ext == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _load_columns_from_file(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old vs. new cleaned outputs.")
    parser.add_argument("old_path", type=Path, help="Path to old dataset (parquet or csv).")
    parser.add_argument("new_path", type=Path, help="Path to new dataset (parquet or csv).")
    parser.add_argument(
        "diff_out",
        type=Path,
        nargs="?",
        default=Path("diff_rows.csv"),
        help="Where to write differing rows as CSV (default: diff_rows.csv).",
    )
    parser.add_argument(
        "--column",
        dest="columns",
        action="extend",
        nargs="+",
        default=[],
        help="Columns to compare (space-separated; repeatable). If omitted, compares all shared non-key columns.",
    )
    parser.add_argument(
        "--columns-file",
        type=Path,
        help="Optional text file (one column name per line) specifying columns to compare.",
    )
    args = parser.parse_args()

    old_df = _read_table(args.old_path)
    new_df = _read_table(args.new_path)

    key_cols = ["assay_id", "smiles"]
    for col in key_cols:
        if col not in old_df.columns or col not in new_df.columns:
            raise KeyError(f"Both files must include '{col}' columns.")

    requested_cols: list[str] = []
    if args.columns_file:
        requested_cols.extend(_load_columns_from_file(args.columns_file))
    if args.columns:
        requested_cols.extend(args.columns)

    requested_cols = [col for col in requested_cols if col not in key_cols]
    if requested_cols:
        compare_cols = sorted(set(requested_cols) & set(old_df.columns) & set(new_df.columns))
        missing = set(requested_cols) - set(compare_cols)
        if missing:
            print(f"[WARN] Skipping missing columns: {sorted(missing)}")
    else:
        compare_cols = sorted(set(old_df.columns) & set(new_df.columns) - set(key_cols))

    if not compare_cols:
        raise SystemExit("No columns to compare after applying filters.")

    def _aggregate(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
        aggs = [pl.len().alias(f"row_count{suffix}")]
        for col in compare_cols:
            aggs.append(
                pl.col(col)
                .drop_nulls()
                .unique()
                .sort()
                .alias(f"{col}{suffix}")
            )
        return df.group_by(key_cols).agg(aggs)

    old_grouped = _aggregate(old_df, "__old")
    new_grouped = _aggregate(new_df, "__new")

    joined = old_grouped.join(new_grouped, on=key_cols, how="outer")

    compare_fields = ["row_count"]
    compare_fields.extend(compare_cols)

    def _ensure_fields(df: pl.DataFrame) -> pl.DataFrame:
        cols = list(df.columns)
        for base in compare_fields:
            old_alias = f"{base}__old"
            new_alias = f"{base}__new"
            if old_alias not in cols:
                df = df.with_columns(pl.lit(None).alias(old_alias))
            if new_alias not in cols:
                df = df.with_columns(pl.lit(None).alias(new_alias))
        return df

    joined = _ensure_fields(joined)

    compare_exprs = []
    for base in compare_fields:
        old_alias = f"{base}__old"
        new_alias = f"{base}__new"
        compare_exprs.append(
            pl.when(pl.col(old_alias).is_null() & pl.col(new_alias).is_null())
            .then(True)
            .otherwise(pl.col(old_alias) == pl.col(new_alias))
        )

    diff_df = joined.filter(~pl.all_horizontal(compare_exprs))
    diff_df.write_csv(args.diff_out)
    print(f"Wrote {diff_df.height} differing rows to {args.diff_out}")


if __name__ == "__main__":
    main()

"""SQLite-backed storage for assay-level metadata."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from .metadata import AssayMetadata


def _connect(db_path: Path) -> sqlite3.Connection:
    """Return a connection to the SQLite database, creating schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assay_metadata (
            aid INTEGER PRIMARY KEY,
            target_type TEXT,
            bioactivity_type TEXT,
            assay_format TEXT,
            selected_column TEXT,
            median REAL,
            mad REAL,
            compounds_screened INTEGER,
            coverage REAL,
            hits_rscore INTEGER,
            hits_overlap INTEGER,
            hits_outcome INTEGER,
            rscore_hit_rate REAL
        )
        """
    )
    conn.commit()
    return conn


def upsert_static_metadata(db_path: Path, rows: Iterable[AssayMetadata]) -> None:
    """Insert or update static fields (labels/format) for each assay.

    This keeps any existing selection/statistics intact.
    """
    conn = _connect(db_path)
    try:
        sql = """
        INSERT INTO assay_metadata (aid, target_type, bioactivity_type, assay_format)
        VALUES (:aid, :target_type, :bioactivity_type, :assay_format)
        ON CONFLICT(aid) DO UPDATE SET
            target_type = excluded.target_type,
            bioactivity_type = excluded.bioactivity_type,
            assay_format = excluded.assay_format
        """
        conn.executemany(
            sql,
            (
                {
                    "aid": row.aid,
                    "target_type": row.target_type,
                    "bioactivity_type": row.bioactivity_type,
                    "assay_format": row.assay_format,
                }
                for row in rows
            ),
        )
        conn.commit()
    finally:
        conn.close()


def import_metadata_csv(db_path: Path, csv_path: Path) -> None:
    """Import an existing metadata CSV into the SQLite store (one-time migration).

    Existing rows in the DB are replaced by the CSV contents.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df = pl.read_csv(csv_path)
    expected_cols = {
        "aid",
        "target_type",
        "bioactivity_type",
        "assay_format",
        "selected_column",
        "median",
        "mad",
        "compounds_screened",
        "coverage",
        "hits_rscore",
        "hits_overlap",
        "hits_outcome",
        "rscore_hit_rate",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM assay_metadata")
        sql = """
        INSERT INTO assay_metadata (
            aid, target_type, bioactivity_type, assay_format,
            selected_column, median, mad, compounds_screened,
            coverage, hits_rscore, hits_overlap, hits_outcome, rscore_hit_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for row in df.to_dicts():
            conn.execute(
                sql,
                (
                    int(row["aid"]),
                    row.get("target_type"),
                    row.get("bioactivity_type"),
                    row.get("assay_format"),
                    row.get("selected_column"),
                    row.get("median"),
                    row.get("mad"),
                    row.get("compounds_screened"),
                    row.get("coverage"),
                    row.get("hits_rscore"),
                    row.get("hits_overlap"),
                    row.get("hits_outcome"),
                    row.get("rscore_hit_rate"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_metadata_stats(
    db_path: Path,
    *,
    aid: int,
    selected_column: str,
    median: float | None,
    mad: float | None,
    compounds_screened: int | None,
    coverage: float | None,
    hits_rscore: int | None,
    hits_overlap: int | None,
    hits_outcome: int | None,
    rscore_hit_rate: float | None,
) -> None:
    """Update selection/statistics fields for a single assay."""
    conn = _connect(db_path)
    try:
        # Ensure row exists
        conn.execute(
            "INSERT INTO assay_metadata (aid) VALUES (?) ON CONFLICT(aid) DO NOTHING",
            (aid,),
        )
        sql = """
        UPDATE assay_metadata
        SET
            selected_column = ?,
            median = ?,
            mad = ?,
            compounds_screened = ?,
            coverage = ?,
            hits_rscore = ?,
            hits_overlap = ?,
            hits_outcome = ?,
            rscore_hit_rate = ?
        WHERE aid = ?
        """
        conn.execute(
            sql,
            (
                selected_column,
                median,
                mad,
                compounds_screened,
                coverage,
                hits_rscore,
                hits_overlap,
                hits_outcome,
                rscore_hit_rate,
                aid,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def iter_selected_for_stats(db_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (aid, selected_column) for assays with a valid selection."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT aid, selected_column
            FROM assay_metadata
            WHERE selected_column IS NOT NULL
              AND selected_column != ''
              AND selected_column != '__INELIGIBLE__'
            """
        )
        for row in cur:
            yield int(row["aid"]), str(row["selected_column"])
    finally:
        conn.close()


def iter_assays_with_any_selection(db_path: Path) -> Iterator[int]:
    """Yield AIDs that have any non-empty selection recorded.

    This includes assays marked as ineligible (selected_column = '__INELIGIBLE__'),
    since they have already been reviewed during column selection.
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT aid
            FROM assay_metadata
            WHERE selected_column IS NOT NULL
              AND selected_column != ''
            """
        )
        for row in cur:
            yield int(row["aid"])
    finally:
        conn.close()


def iter_selected_for_rscores(
    db_path: Path,
) -> Iterator[tuple[int, str, float | None, float | None]]:
    """Yield (aid, selected_column, median, mad) for r-score computation."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT aid, selected_column, median, mad
            FROM assay_metadata
            WHERE selected_column IS NOT NULL
              AND selected_column != ''
              AND selected_column != '__INELIGIBLE__'
            """
        )
        for row in cur:
            yield (
                int(row["aid"]),
                str(row["selected_column"]),
                row["median"],
                row["mad"],
            )
    finally:
        conn.close()


def export_metadata_to_csv(db_path: Path, csv_path: Path) -> None:
    """Export the entire metadata table to a CSV file."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                aid,
                target_type,
                bioactivity_type,
                assay_format,
                selected_column,
                median,
                mad,
                compounds_screened,
                coverage,
                hits_rscore,
                hits_overlap,
                hits_outcome,
                rscore_hit_rate
            FROM assay_metadata
            ORDER BY aid
            """
        )
        rows = [dict(row) for row in cur]
    finally:
        conn.close()

    if not rows:
        df = pl.DataFrame(
            [
                asdict(
                    AssayMetadata(
                        aid=0,
                        target_type=None,
                        bioactivity_type=None,
                        assay_format=None,
                        selected_column=None,
                        median=None,
                        mad=None,
                        compounds_screened=None,
                        coverage=None,
                        hits_rscore=None,
                        hits_overlap=None,
                        hits_outcome=None,
                        rscore_hit_rate=None,
                    )
                )
            ]
        ).head(0)
    else:
        # Increase infer_schema_length to cover all rows so mixed null/non-null
        # columns get a consistent inferred dtype.
        df = pl.DataFrame(rows, infer_schema_length=len(rows))

    # Rename the primary key column to match downstream pipelines.
    if "aid" in df.columns:
        df = df.rename({"aid": "assay_id"})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv_path)

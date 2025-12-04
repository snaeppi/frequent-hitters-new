"""Assay-level metadata construction and update."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .pcassay_catalog import iter_aids_from_pcassay


@dataclass(slots=True)
class AssayMetadata:
    aid: int
    target_type: str | None
    bioactivity_type: str | None
    assay_format: str | None
    selected_column: str | None
    median: float | None
    mad: float | None
    compounds_screened: int | None
    coverage: float | None
    hits_rscore: int | None
    hits_overlap: int | None
    hits_outcome: int | None
    rscore_hit_rate: float | None


def _labels_from_hit_dexter_single(
    dataset_label: str | None,
) -> tuple[str, str | None] | None:
    """Match the original single-label mapping from Hit Dexter."""
    if dataset_label is None:
        return None
    label = dataset_label.strip().lower()
    if label == "target based":
        return ("target-based", "specific bioactivity")
    if label == "cell based":
        return ("cell-based", "specific bioactivity")
    if label == "extended cell based":
        return ("cell-based", None)
    return None


def _labels_from_label_set(
    labels: set[str] | None,
) -> tuple[str, str | None] | None:
    """Derive (target_type, bioactivity_type) from a set of Hit Dexter labels.

    Preference order:
    1) target based
    2) cell based
    3) extended cell based
    """
    if not labels:
        return None
    lowered = {lbl.strip().lower() for lbl in labels if lbl}
    if "target based" in lowered:
        return _labels_from_hit_dexter_single("target based")
    if "cell based" in lowered:
        return _labels_from_hit_dexter_single("cell based")
    if "extended cell based" in lowered:
        return _labels_from_hit_dexter_single("extended cell based")
    # Fallback: pick an arbitrary label and try the single-label mapping.
    for lbl in lowered:
        mapped = _labels_from_hit_dexter_single(lbl)
        if mapped is not None:
            return mapped
    return None


def _format_from_labels(
    target_type: str | None,
    bioactivity_type: str | None,
    has_annotation: bool,
) -> str | None:
    """Compute assay_format from target/bioactivity and annotation presence.

    Rules:
    1) biochemical iff target-based AND specific bioactivity
    2) cellular iff cell-based AND specific bioactivity
    3) other for any other *annotated* case
    4) empty (None) when there is no annotation
    """
    if not has_annotation:
        return None
    if target_type == "target-based" and bioactivity_type == "specific bioactivity":
        return "biochemical"
    if target_type == "cell-based" and bioactivity_type == "specific bioactivity":
        return "cellular"
    return "other"


def _load_hd3_annotations(path: Path) -> dict[int, set[str]]:
    """Return mapping AID -> set of dataset labels (to handle duplicates)."""
    if not path.exists():
        raise FileNotFoundError(f"hd3_annotations.csv not found: {path}")
    df = pl.read_csv(path, separator=";")
    # Expect columns: "data set", "AID", ...
    if "AID" not in df.columns or "data set" not in df.columns:
        raise ValueError("Unexpected hd3_annotations.csv schema.")
    records: dict[int, set[str]] = {}
    for row in df.select("AID", "data set").iter_rows(named=True):
        try:
            aid = int(row["AID"])
        except (TypeError, ValueError):
            continue
        dataset = row["data set"]
        if isinstance(dataset, str):
            records.setdefault(aid, set()).add(dataset)
    return records


def build_initial_metadata(
    *,
    pcassay_result: Path,
    hd3_annotations: Path,
    aids: Iterable[int] | None = None,
) -> pl.DataFrame:
    """Build initial metadata with empty selection/stats."""
    annotations = _load_hd3_annotations(hd3_annotations)
    if aids is None:
        aids = iter_aids_from_pcassay(pcassay_result)
    aid_list = sorted({int(a) for a in aids})

    rows: list[AssayMetadata] = []
    for aid in aid_list:
        label_set = annotations.get(aid)
        labels = _labels_from_label_set(label_set)
        if labels is None:
            target_type = None
            bioactivity_type = None
        else:
            target_type, bioactivity_type = labels
        assay_format = _format_from_labels(
            target_type=target_type,
            bioactivity_type=bioactivity_type,
            has_annotation=label_set is not None,
        )
        rows.append(
            AssayMetadata(
                aid=aid,
                target_type=target_type,
                bioactivity_type=bioactivity_type,
                assay_format=assay_format,
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

    if not rows:
        return pl.DataFrame(
            {
                "aid": pl.Series([], dtype=pl.Int64),
                "target_type": pl.Series([], dtype=pl.String),
                "bioactivity_type": pl.Series([], dtype=pl.String),
                "assay_format": pl.Series([], dtype=pl.String),
                "selected_column": pl.Series([], dtype=pl.String),
                "median": pl.Series([], dtype=pl.Float64),
                "mad": pl.Series([], dtype=pl.Float64),
                "compounds_screened": pl.Series([], dtype=pl.Int64),
                "coverage": pl.Series([], dtype=pl.Float64),
                "hits_rscore": pl.Series([], dtype=pl.Int64),
                "hits_overlap": pl.Series([], dtype=pl.Int64),
                "hits_outcome": pl.Series([], dtype=pl.Int64),
                "rscore_hit_rate": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame([asdict(row) for row in rows])


def update_metadata_row_with_stats(
    *,
    metadata_db: Path,
    aid: int,
    selected_column: str,
    median: float | None,
    mad: float | None,
    compounds_screened: int | None,
    coverage: float | None,
    hits_rscore: int | None,
    hits_overlap: int | None,
    hits_outcome: int | None,
) -> None:
    """Update a single row in the SQLite metadata store with selection + stats."""
    from .meta_db import update_metadata_stats

    if hits_rscore is not None and compounds_screened:
        rate_value: float | None = hits_rscore / compounds_screened
    else:
        rate_value = None

    update_metadata_stats(
        metadata_db,
        aid=aid,
        selected_column=selected_column,
        median=median,
        mad=mad,
        compounds_screened=compounds_screened,
        coverage=coverage,
        hits_rscore=hits_rscore,
        hits_overlap=hits_overlap,
        hits_outcome=hits_outcome,
        rscore_hit_rate=rate_value,
    )

"""Interactive column selection using r-score based statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, cast

import polars as pl
from rich.console import Console
from rich.table import Table

from .metadata import update_metadata_row_with_stats
from .polars_helpers import is_numeric_dtype, numeric_expr
from .rich_console import console

OUTCOME_COLUMN = "PUBCHEM_ACTIVITY_OUTCOME"
CID_COLUMN = "PUBCHEM_CID"
EXCLUDED_COLUMNS = {CID_COLUMN, OUTCOME_COLUMN}
ASSAY_ID_PLACEHOLDER = "assay_id"
INELIGIBLE_SENTINEL = "__INELIGIBLE__"


@dataclass(slots=True)
class CandidateStats:
    column_name: str
    median: float | None
    mad: float | None
    compounds_screened: int
    coverage: float | None
    hits_rscore: int | None
    hits_overlap: int | None
    hits_outcome: int | None


def _replicate_base(name: str) -> str | None:
    """Return a base name if the column looks like a replicate."""
    import re

    # Leading replicate token: REPLICATE_A_xxx or REP_A_xxx or TRIAL_1_xxx
    m_lead = re.match(
        r"^(?:replicate|rep|trial)[-_ ]?([A-Za-z0-9])[_-]?(.*)$",
        name,
        re.IGNORECASE,
    )
    if m_lead:
        rest = m_lead.group(2).lstrip("_- ")
        return rest or None

    # Anywhere in the middle: prefix + token + tag + rest
    m_mid = re.match(
        r"^(.*?)(?:replicate|rep|trial)[-_ ]?([A-Za-z0-9])[_-]?(.*)$",
        name,
        re.IGNORECASE,
    )
    if m_mid:
        prefix = (m_mid.group(1) or "").rstrip("_- ")
        rest = (m_mid.group(3) or "").lstrip("_- ")
        base = f"{prefix}_{rest}".strip("_- ")
        return base or None

    # Suffix replicate token: base_A, base_B, base_1, etc.
    # Example: Z-score_A, Z-score_B -> base 'Z-score'
    m_suffix = re.match(r"^(.*?)[_-]([A-Za-z0-9])$", name)
    if m_suffix:
        prefix = (m_suffix.group(1) or "").rstrip("_- ")
        base = prefix.strip("_- ")
        return base or None

    return None


def _candidate_columns(schema: dict[str, pl.DataType]) -> List[Tuple[str, pl.DataType]]:
    candidates: List[Tuple[str, pl.DataType]] = []
    for name, dtype in schema.items():
        if name in EXCLUDED_COLUMNS:
            continue
        upper = name.upper()
        if upper.startswith("REPLICATE") or upper.startswith("REP_") or upper.startswith("TRIAL"):
            # Skip raw replicate members; we prefer mean columns.
            continue
        if name.startswith("PUBCHEM") and name not in {"PUBCHEM_ACTIVITY_SCORE"}:
            # Keep score, drop other PUBCHEM_* metadata columns.
            continue
        if is_numeric_dtype(dtype):
            candidates.append((name, cast(pl.DataType, dtype)))
    return candidates


def _collect_scalar(lf: pl.LazyFrame, expr: pl.Expr, alias: str) -> float | None:
    frame = lf.select(expr.alias(alias)).collect(engine="streaming")
    series = frame.get_column(alias)
    if series.is_null().all():
        return None
    return float(series.item())


def _compute_candidate_stats(
    *,
    assay_parquet: Path,
) -> Tuple[int, List[CandidateStats]]:
    """Return (row_count, candidate stats) for a per-assay table parquet."""
    if not assay_parquet.exists():
        raise FileNotFoundError(f"Assay table parquet not found: {assay_parquet}")

    lf = pl.scan_parquet(assay_parquet)
    schema = lf.collect_schema()
    if OUTCOME_COLUMN not in schema:
        raise ValueError(f"Assay table parquet missing {OUTCOME_COLUMN}")
    if CID_COLUMN not in schema:
        raise ValueError(f"Assay table parquet missing {CID_COLUMN}")

    # Detect replicate groups among numeric candidates and add mean columns,
    # persisted back to the per-assay table parquet. This ensures that challenging
    # replicate means (e.g., _ACTIVITY_SCORE_12.5uM_(%)_mean) exist.
    initial_numeric: list[tuple[str, pl.DataType]] = []
    for name, dtype in schema.items():
        if name in EXCLUDED_COLUMNS:
            continue
        if not is_numeric_dtype(dtype):
            continue
        if _replicate_base(name) is not None:
            initial_numeric.append((name, dtype))
    replicate_groups: dict[str, list[str]] = {}
    for name, _dtype in initial_numeric:
        base = _replicate_base(name)
        if base:
            replicate_groups.setdefault(base, []).append(name)
    # Treat any detected replicate group (including a single surviving member)
    # as eligible for a mean column. This ensures that datasets where one
    # replicate was dropped upstream still get a stable `_..._mean` name.
    replicate_means = replicate_groups
    if replicate_means:
        new_cols: list[pl.Expr] = []
        for base, cols in replicate_means.items():
            # Match historical naming: leading underscore + base + _mean.
            new_name = f"_{base}_mean"
            value_exprs = [pl.col(c).cast(pl.Float64) for c in cols]
            mean_expr = pl.mean_horizontal(value_exprs).alias(new_name)
            new_cols.append(mean_expr)
        if new_cols:
            lf_with_means = lf.with_columns(new_cols)
            tmp_path = assay_parquet.with_suffix(".tmp.parquet")
            lf_with_means.sink_parquet(tmp_path, compression="zstd")
            tmp_path.replace(assay_parquet)
            lf = pl.scan_parquet(assay_parquet)
            schema = lf.collect_schema()

    row_count = (
        lf.select(pl.len().alias("row_count"))
        .collect(engine="streaming")
        .get_column("row_count")
        .item()
    )
    if row_count == 0:
        return 0, []

    # Candidate columns: numeric types plus string-like columns that can be
    # sensibly cast to Float64 (so we don't drop depositor numeric columns
    # mis-inferred as strings by Polars).
    candidates: List[Tuple[str, pl.DataType]] = []
    for name, dtype in schema.items():
        if name in EXCLUDED_COLUMNS:
            continue
        upper = name.upper()
        if upper.startswith("REPLICATE") or upper.startswith("REP_") or upper.startswith("TRIAL"):
            # Skip raw replicate members; we prefer mean columns.
            continue
        if name.startswith("PUBCHEM") and name not in {"PUBCHEM_ACTIVITY_SCORE"}:
            # Keep score, drop other PUBCHEM_* metadata columns.
            continue
        if is_numeric_dtype(dtype):
            candidates.append((name, dtype))
        else:
            # Try to treat convertible string columns as numeric.
            cast_expr = pl.col(name).cast(pl.Float64, strict=False)
            nonnull = (
                lf.select(cast_expr.is_not_null().sum().alias("__nonnull_tmp"))
                .collect(engine="streaming")
                .get_column("__nonnull_tmp")
                .item()
            )
            if nonnull > 0:
                candidates.append((name, cast(pl.DataType, pl.Float64)))

    stats: List[CandidateStats] = []

    for name, dtype in candidates:
        base_expr = numeric_expr(name, dtype)
        median = _collect_scalar(lf, base_expr.median(), "median")
        if median is None:
            stats.append(
                CandidateStats(
                    column_name=name,
                    median=None,
                    mad=None,
                    compounds_screened=row_count,
                    coverage=None,
                    hits_rscore=None,
                    hits_overlap=None,
                    hits_outcome=None,
                )
            )
            continue
        mad = _collect_scalar(
            lf,
            (base_expr - pl.lit(median)).abs().median(),
            "mad",
        )
        if mad is None or mad == 0:
            stats.append(
                CandidateStats(
                    column_name=name,
                    median=median,
                    mad=mad,
                    compounds_screened=row_count,
                    coverage=None,
                    hits_rscore=None,
                    hits_overlap=None,
                    hits_outcome=None,
                )
            )
            continue

        coverage = (
            lf.select(base_expr.is_not_null().sum().alias("nonnull"))
            .collect(engine="streaming")
            .get_column("nonnull")
            .item()
        )
        coverage_ratio = coverage / row_count if row_count else None

        scaled_mad = mad * 1.4826
        r_expr = ((base_expr - pl.lit(median)) / pl.lit(scaled_mad)).alias("r_score")
        stats_df = (
            lf.select(
                r_expr,
                (pl.col(OUTCOME_COLUMN) == "Active").cast(pl.Int8).alias("__gt_outcome"),
            )
            .drop_nulls(["r_score", "__gt_outcome"])
            .select(
                (pl.col("r_score").abs() >= 3.0).sum().alias("hit_count_r"),
                ((pl.col("r_score").abs() >= 3.0) & (pl.col("__gt_outcome") == 1))
                .sum()
                .alias("hit_overlap"),
                (pl.col("__gt_outcome") == 1).sum().alias("hit_outcome"),
            )
            .collect(engine="streaming")
        )
        hits_rscore = int(stats_df.get_column("hit_count_r").item())
        hits_overlap = int(stats_df.get_column("hit_overlap").item())
        hits_outcome = int(stats_df.get_column("hit_outcome").item())

        stats.append(
            CandidateStats(
                column_name=name,
                median=median,
                mad=mad,
                compounds_screened=row_count,
                coverage=coverage_ratio,
                hits_rscore=hits_rscore,
                hits_overlap=hits_overlap,
                hits_outcome=hits_outcome,
            )
        )

    return row_count, stats


def interactive_select_for_aid(
    *,
    aid: int,
    assay_parquet: Path,
    metadata_db: Path,
    io_console: Console | None = None,
) -> str | None:
    """Interactive column selection for a single AID.

    Expects the per-assay table parquet generated by download-assays (aid_<AID>.parquet).
    """
    if not assay_parquet.exists():
        raise FileNotFoundError(
            f"Assay table parquet not found for AID {aid}: {assay_parquet} "
            "(run download-assays first)."
        )

    io = io_console or console

    row_count, candidates = _compute_candidate_stats(assay_parquet=assay_parquet)
    if not candidates:
        io.print(f"AID {aid}: no candidate columns.")
        return None

    # Determine the best candidate by (hits_overlap, coverage).
    best_idx = None
    best_overlap = -1
    best_cov = -1.0
    for idx, c in enumerate(candidates):
        overlap = c.hits_overlap or 0
        cov = c.coverage or 0.0
        if overlap > best_overlap or (overlap == best_overlap and cov > best_cov):
            best_idx = idx
            best_overlap = overlap
            best_cov = cov

    table = Table(title=f"AID {aid} candidates")
    table.add_column("Idx", justify="right")
    table.add_column("Column")
    table.add_column("# r-score hits", justify="right")
    table.add_column("# overlap hits", justify="right")
    table.add_column("# outcome hits", justify="right")
    table.add_column("Coverage", justify="right")

    for idx, c in enumerate(candidates):
        cov_str = f"{c.coverage:.1%}" if c.coverage is not None else "n/a"
        row_style = "bold green" if idx == best_idx else None
        table.add_row(
            str(idx),
            c.column_name,
            str(c.hits_rscore or 0),
            str(c.hits_overlap or 0),
            str(c.hits_outcome or 0),
            cov_str,
            style=row_style,
        )

    io.print(f"Screens on assay: {row_count}")
    io.print(table)

    best_label = ""
    if best_idx is not None:
        best_label = f"[green]Enter = best overlap ({candidates[best_idx].column_name})[/green]"
    prompt_text = (
        f"Select index ({best_label or 'Enter = best overlap'}, "
        "[yellow]'s'[/yellow]=skip, [red]'i'[/red]=mark ineligible): "
    )

    choice = io.input(prompt_text).strip()
    lower = choice.lower()
    if lower.startswith("s"):
        io.print("Skipped selection.")
        return None
    if lower.startswith("i"):
        update_metadata_row_with_stats(
            metadata_db=metadata_db,
            aid=aid,
            selected_column=INELIGIBLE_SENTINEL,
            median=None,
            mad=None,
            compounds_screened=row_count,
            coverage=None,
            hits_rscore=None,
            hits_overlap=None,
            hits_outcome=None,
        )
        return INELIGIBLE_SENTINEL

    selected_idx = best_idx
    if choice:
        try:
            idx_val = int(choice)
            if 0 <= idx_val < len(candidates):
                selected_idx = idx_val
        except ValueError:
            selected_idx = best_idx

    if selected_idx is None:
        io.print("No valid selection made.")
        return None

    selected = candidates[selected_idx]
    cov_desc = f"{selected.coverage:.1%}" if selected.coverage is not None else "n/a"
    update_metadata_row_with_stats(
        metadata_db=metadata_db,
        aid=aid,
        selected_column=selected.column_name,
        median=selected.median,
        mad=selected.mad,
        compounds_screened=selected.compounds_screened,
        coverage=selected.coverage,
        hits_rscore=selected.hits_rscore,
        hits_overlap=selected.hits_overlap,
        hits_outcome=selected.hits_outcome,
    )
    io.print(
        f"Selected column: {selected.column_name} "
        f"(hits_rscore={selected.hits_rscore or 0}, "
        f"overlap={selected.hits_overlap or 0}/{selected.hits_outcome or 0}, "
        f"coverage={cov_desc})"
    )
    return selected.column_name

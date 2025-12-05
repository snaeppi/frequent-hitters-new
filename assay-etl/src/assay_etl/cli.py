"""Typer-based CLI entry point for the PubChem bioassay preprocessing tool."""

from __future__ import annotations

import httpx
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .download import download_assays
from .meta_db import (
    export_metadata_to_csv,
    import_metadata_csv,
    iter_selected_for_stats,
    upsert_static_metadata,
)
from .metadata import AssayMetadata, build_initial_metadata, update_metadata_row_with_stats
from .pcassay_catalog import iter_aids_from_pcassay
from .rscores import compute_rscores_from_metadata
from .selection import _compute_candidate_stats, interactive_select_for_aid
from .manual_annotation import (
    annotate_metadata_manual,
    iter_annotation_contexts,
    load_annotation_context,
)

console: Console = Console()

PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
PROGRESS_OPTS = {"console": console, "refresh_per_second": 0}

app = typer.Typer(
    no_args_is_help=True,
    help="File-based tooling for PubChem bioassay tables.",
)


def _read_aids_file(path: Path) -> List[int]:
    aids: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                aids.append(int(line))
            except ValueError:
                continue
    return aids


def _resolve_aids(
    *,
    aid: Optional[int],
    from_pcassay: bool,
    aids_file: Optional[Path],
    pcassay_result: Path,
) -> List[int]:
    if aids_file is not None:
        return _read_aids_file(aids_file)
    if aid is not None:
        return [aid]
    if from_pcassay:
        return sorted(iter_aids_from_pcassay(pcassay_result))
    raise typer.BadParameter("Specify --aid, --from-pcassay, or --aids-file.")


def _aggregated_path(base_dir: Path, aid: int) -> Path:
    """Prefer the new naming (aid_<AID>.parquet), fall back to legacy suffix."""
    preferred = base_dir / f"aid_{aid}.parquet"
    legacy = base_dir / f"aid_{aid}_cid_agg.parquet"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


@app.command("download-assays")
def download_assays_command(
    aid: Optional[int] = typer.Option(
        None,
        "--aid",
        help="Single AID to download.",
    ),
    from_pcassay: bool = typer.Option(
        False,
        "--from-pcassay",
        help="Use all AIDs discovered from pcassay_result.txt.",
    ),
    aids_file: Optional[Path] = typer.Option(
        None,
        "--aids-file",
        help="Optional file containing one AID per line.",
    ),
    pcassay_result: Path = typer.Option(
        Path("data/pcassay_result.txt"),
        "--pcassay-result",
        help="Path to pcassay_result.txt catalog export.",
    ),
    out_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--out-dir",
        help="Directory for aggregated per-assay Parquet tables (one row per compound).",
    ),
    force_download: bool = typer.Option(
        False,
        "--force-download/--no-force-download",
        help="Force re-download even if parquet exists.",
    ),
    io_workers: int = typer.Option(
        4,
        "--io-workers",
        help="Maximum concurrent downloads.",
    ),
    cpu_workers: int = typer.Option(
        4,
        "--cpu-workers",
        help="Maximum concurrent processing workers.",
    ),
) -> None:
    """Download and aggregate assay tables from PubChem (one parquet per AID)."""
    aids = _resolve_aids(
        aid=aid,
        from_pcassay=from_pcassay,
        aids_file=aids_file,
        pcassay_result=pcassay_result,
    )
    if not aids:
        typer.echo("No AIDs resolved.", err=True)
        raise typer.Exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = download_assays(
        aids=aids,
        cache_dir=out_dir,
        force_download=force_download,
        io_workers=io_workers,
        cpu_workers=cpu_workers,
    )
    typer.echo(f"Downloaded/verified {len(results)} assays into {out_dir}")


@app.command("summarize-assays")
def summarize_assays_command(
    pcassay_result: Path = typer.Option(
        Path("data/pcassay_result.txt"),
        "--pcassay-result",
        help="Path to pcassay_result.txt catalog export.",
    ),
    hd3_annotations: Path = typer.Option(
        Path("data/hd3_annotations.csv"),
        "--hd3-annotations",
        help="Path to hd3_annotations.csv file.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="SQLite database for assay-level metadata.",
    ),
    metadata_csv: Path = typer.Option(
        Path("outputs/assay_metadata.csv"),
        "--metadata-csv",
        help="Optional CSV export of assay-level metadata.",
    ),
) -> None:
    """Create or refresh assay metadata in SQLite (labels + formats)."""
    df = build_initial_metadata(
        pcassay_result=pcassay_result,
        hd3_annotations=hd3_annotations,
        aids=None,
    )
    rows = [AssayMetadata(**row) for row in df.to_dicts()]
    upsert_static_metadata(metadata_db, rows)
    typer.echo(f"Upserted metadata for {df.height} assays into {metadata_db}")
    if metadata_csv:
        export_metadata_to_csv(metadata_db, metadata_csv)
        typer.echo(f"Exported metadata snapshot to {metadata_csv}")


@app.command("select-column")
def select_column_command(
    aid: Optional[int] = typer.Option(
        None,
        "--aid",
        help="Annotate a single assay; use --from-pcassay/--aids-file for batch.",
    ),
    from_pcassay: bool = typer.Option(
        False,
        "--from-pcassay",
        help="Iterate over AIDs from pcassay_result.txt (or --aids-file).",
    ),
    aids_file: Optional[Path] = typer.Option(
        None,
        "--aids-file",
        help="Optional file containing one AID per line.",
    ),
    pcassay_result: Path = typer.Option(
        Path("data/pcassay_result.txt"),
        "--pcassay-result",
        help="Path to pcassay_result.txt catalog export.",
    ),
    start_aid: Optional[int] = typer.Option(
        None,
        "--start-aid",
        help="Optional minimum AID; skip anything below this when batching.",
    ),
    assays_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--assays-dir",
        "--aggregated-dir",
        help="Directory containing aggregated per-assay parquet tables (aid_<AID>.parquet).",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="SQLite metadata database to update with selected column and stats.",
    ),
) -> None:
    """Interactively pick a primary screening column for one or many assays."""
    if aid is not None and (from_pcassay or aids_file is not None):
        raise typer.BadParameter("Use either --aid or batching flags, not both.")
    if aid is None and not from_pcassay and aids_file is None:
        raise typer.BadParameter("Specify --aid or use --from-pcassay/--aids-file.")

    aids = _resolve_aids(
        aid=aid,
        from_pcassay=from_pcassay,
        aids_file=aids_file,
        pcassay_result=pcassay_result,
    )
    if start_aid is not None:
        aids = [a for a in aids if a >= start_aid]
    if not aids:
        typer.echo("No AIDs resolved.")
        return

    console.line()  # Spacer above progress bar
    with Progress(*PROGRESS_COLUMNS, **PROGRESS_OPTS) as progress:
        task_id = progress.add_task("Selecting columns", total=len(aids))
        for current_aid in aids:
            progress.update(task_id, description=f"AID {current_aid}")

            try:
                result = interactive_select_for_aid(
                    aid=current_aid,
                    aggregated_parquet=_aggregated_path(assays_dir, current_aid),
                    metadata_db=metadata_db,
                )
            except FileNotFoundError as exc:
                typer.echo(str(exc))
                progress.advance(task_id)
                continue
            except KeyboardInterrupt:
                progress.stop()
                typer.echo("\nStopping selection loop on user interrupt.")
                break
            if result == "__INELIGIBLE__":
                typer.echo("Marked as ineligible.")
            elif result:
                typer.echo(f"Selection recorded for AID {current_aid}: {result}")
            else:
                typer.echo("No selection recorded for this AID (skipped).")

            progress.advance(task_id)
        progress.update(task_id, description="[green]All selections processed")
    console.line()  # Spacer below progress bar


@app.command("compute-rscores")
def compute_rscores_command(
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="SQLite metadata database containing selected columns and stats.",
    ),
    aggregated_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--aggregated-dir",
        "--assays-dir",
        help="Directory containing aggregated per-assay parquet tables (aid_<AID>.parquet).",
    ),
    out_parquet: Path = typer.Option(
        Path("outputs/assay_rscores.parquet"),
        "--out",
        help="Destination parquet for combined r-scores.",
    ),
) -> None:
    """Compute r-scores for all assays with a selected column."""
    compute_rscores_from_metadata(
        metadata_db=metadata_db,
        aggregated_dir=aggregated_dir,
        output_parquet=out_parquet,
    )
    typer.echo(f"Wrote r-score table to {out_parquet}")


@app.command("update-selected-stats")
def update_selected_stats_command(
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Metadata SQLite DB containing selected columns (will be updated with stats).",
    ),
    aggregated_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--aggregated-dir",
        "--assays-dir",
        help="Directory containing aggregated per-assay parquet tables (aid_<AID>.parquet).",
    ),
) -> None:
    """Compute stats for already-selected columns and update metadata DB."""

    rows = list(iter_selected_for_stats(metadata_db))
    if not rows:
        typer.echo("No assays with selected_column to update.")
        return

    updated = 0
    console.line()  # Spacer above progress bar
    with Progress(*PROGRESS_COLUMNS, **PROGRESS_OPTS) as progress:
        task_id = progress.add_task("Updating stats", total=len(rows))
        for aid, selected_column in rows:
            progress.update(task_id, description=f"AID {aid}")
            agg_path = _aggregated_path(aggregated_dir, aid)
            if not agg_path.exists():
                typer.echo(f"[skip] Aggregated parquet missing for AID {aid}: {agg_path}")
                progress.advance(task_id)
                continue
            row_count, stats = _compute_candidate_stats(aggregated_parquet=agg_path)
            match = next((c for c in stats if c.column_name == selected_column), None)
            if match is None:
                typer.echo(
                    f"[warn] Selected column '{selected_column}' not found in candidates for AID {aid}."
                )
                progress.advance(task_id)
                continue
            update_metadata_row_with_stats(
                metadata_db=metadata_db,
                aid=aid,
                selected_column=match.column_name,
                median=match.median,
                mad=match.mad,
                compounds_screened=match.compounds_screened,
                coverage=match.coverage,
                hits_rscore=match.hits_rscore,
                hits_overlap=match.hits_overlap,
                hits_outcome=match.hits_outcome,
            )
            updated += 1
            progress.advance(task_id)
        progress.update(task_id, description="[green]Stats update complete")
    console.line()  # Spacer below progress bar

    typer.echo(f"Updated stats for {updated} assays in {metadata_db}")


@app.command("import-metadata-csv")
def import_metadata_csv_command(
    csv_path: Path = typer.Argument(
        Path("outputs/assay_metadata.csv"),
        help="Existing metadata CSV to import.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Destination metadata SQLite database.",
    ),
) -> None:
    """Import an existing assay_metadata.csv into the SQLite metadata DB."""
    import_metadata_csv(metadata_db, csv_path)
    typer.echo(f"Imported metadata from {csv_path} into {metadata_db}")


@app.command("export-metadata-csv")
def export_metadata_csv_command(
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Metadata SQLite DB to export from.",
    ),
    out_csv: Path = typer.Option(
        Path("outputs/assay_metadata.csv"),
        "--out",
        help="Destination metadata CSV snapshot.",
    ),
) -> None:
    """Export the current metadata table from SQLite to CSV."""
    export_metadata_to_csv(metadata_db, out_csv)
    typer.echo(f"Exported metadata snapshot from {metadata_db} to {out_csv}")


@app.command("annotate-metadata")
def annotate_metadata_command(
    aid: Optional[int] = typer.Option(
        None,
        "--aid",
        help="Annotate a single AID (overrides existing labels if present).",
    ),
    from_pcassay: bool = typer.Option(
        False,
        "--from-pcassay",
        help="Annotate eligible assays from pcassay_result.txt (or --aids-file).",
    ),
    aids_file: Optional[Path] = typer.Option(
        None,
        "--aids-file",
        help="Optional file containing one AID per line (use with --from-pcassay).",
    ),
    pcassay_result: Path = typer.Option(
        Path("data/pcassay_result.txt"),
        "--pcassay-result",
        help="Path to pcassay_result.txt catalog export.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Metadata SQLite DB containing assays and selections.",
    ),
    start_aid: Optional[int] = typer.Option(
        None,
        "--start-aid",
        help="Optional minimum AID; skip anything below this when annotating all.",
    ),
) -> None:
    """Manually annotate target_type/bioactivity_type for assays with selections (use --from-pcassay to iterate all missing)."""
    if aid is not None and from_pcassay:
        raise typer.BadParameter("Use either --aid or --from-pcassay, not both.")
    if aid is None and not from_pcassay and aids_file is None:
        raise typer.BadParameter("Specify --aid or use --from-pcassay/--aids-file to annotate.")

    client = httpx.Client(timeout=60.0)
    try:
        if aid is not None:
            ctx = load_annotation_context(metadata_db=metadata_db, aid=aid, include_annotated=True)
            if ctx is None:
                return
            annotate_metadata_manual(
                metadata_db=metadata_db,
                context=ctx,
                client=client,
                index=1,
                total=1,
            )
            return

        aids = _resolve_aids(
            aid=None,
            from_pcassay=from_pcassay,
            aids_file=aids_file,
            pcassay_result=pcassay_result,
        )
        contexts = iter_annotation_contexts(
            metadata_db=metadata_db,
            aids=aids,
            start_aid=start_aid,
            include_annotated=False,
        )
        if not contexts:
            typer.echo("No assays require manual annotation.")
            return

        console.line()  # Spacer above progress bar
        with Progress(*PROGRESS_COLUMNS, **PROGRESS_OPTS) as progress:
            task_id = progress.add_task("Annotating assays", total=len(contexts))
            for idx, ctx in enumerate(contexts, start=1):
                progress.update(task_id, description=f"Annotating AID {ctx.aid}")
                annotate_metadata_manual(
                    metadata_db=metadata_db,
                    context=ctx,
                    client=client,
                    index=idx,
                    total=len(contexts),
                )
                progress.advance(task_id)
            progress.update(task_id, description="[green]All annotations processed")
        console.line()  # Spacer below progress bar
    finally:
        client.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()

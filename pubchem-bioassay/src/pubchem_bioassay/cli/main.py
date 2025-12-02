"""Typer-based CLI entry point for the PubChem bioassay preprocessing tool."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import typer
from dotenv import load_dotenv

from ..aggregation import aggregate_compounds
from ..download import download_assays
from ..meta_db import (
    export_metadata_to_csv,
    import_metadata_csv,
    iter_selected_for_stats,
    upsert_static_metadata,
)
from ..metadata import AssayMetadata, build_initial_metadata, update_metadata_row_with_stats
from ..pcassay_catalog import iter_aids_from_pcassay
from ..rscores import compute_rscores_from_metadata
from ..selection import _compute_candidate_stats, interactive_select_for_aid
from ..llm_annotation import annotate_metadata_llm

app = typer.Typer(
    no_args_is_help=True,
    help="Minimal, file-based tooling for PubChem bioassay tables.",
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
        help="Directory for per-assay Parquet tables.",
    ),
    force_download: bool = typer.Option(
        False,
        "--force-download/--no-force-download",
        help="Force re-download even if parquet exists.",
    ),
) -> None:
    """Download gzipped CSVs from PubChem and materialize as Parquet."""
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
    results = download_assays(aids=aids, cache_dir=out_dir, force_download=force_download)
    typer.echo(f"Downloaded/verified {len(results)} assays into {out_dir}")


@app.command("aggregate-compounds")
def aggregate_compounds_command(
    aid: Optional[int] = typer.Option(
        None,
        "--aid",
        help="Single AID to aggregate.",
    ),
    from_pcassay: bool = typer.Option(
        False,
        "--from-pcassay",
        help="Aggregate all AIDs discovered from pcassay_result.txt.",
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
    raw_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--raw-dir",
        help="Directory containing raw per-assay parquet tables.",
    ),
    out_dir: Path = typer.Option(
        Path("outputs/aggregated"),
        "--out-dir",
        help="Directory for aggregated per-CID parquet tables.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Recompute even if aggregated parquet already exists.",
    ),
) -> None:
    """Aggregate assay tables to one row per CID."""
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
    skipped = 0
    processed = 0
    for a in aids:
        input_parquet = raw_dir / f"aid_{a}.parquet"
        output_parquet = out_dir / f"aid_{a}_cid_agg.parquet"
        if output_parquet.exists() and not force:
            typer.echo(f"[skip] Aggregated parquet already exists for AID {a}: {output_parquet}")
            skipped += 1
            continue
        aggregate_compounds(aid=a, input_parquet=input_parquet, output_parquet=output_parquet)
        processed += 1
    typer.echo(f"Aggregated {processed} assays into {out_dir} (skipped {skipped} existing files)")


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
    aid: int = typer.Argument(..., help="Assay AID for manual column selection."),
    raw_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--raw-dir",
        help="Directory containing raw per-assay parquet tables.",
    ),
    aggregated_dir: Path = typer.Option(
        Path("outputs/aggregated"),
        "--aggregated-dir",
        help="Directory containing aggregated per-assay parquet tables.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="SQLite metadata database to update with selected column and stats.",
    ),
) -> None:
    """Interactively pick a primary screening column using r-score stats."""
    raw_path = raw_dir / f"aid_{aid}.parquet"
    agg_path = aggregated_dir / f"aid_{aid}_cid_agg.parquet"
    result = interactive_select_for_aid(
        aid=aid,
        aggregated_parquet=agg_path,
        raw_parquet=raw_path,
        metadata_db=metadata_db,
    )
    if result == "__INELIGIBLE__":
        typer.echo("Marked as ineligible.")
    elif result:
        typer.echo(f"Selection recorded for AID {aid}: {result}")
    else:
        typer.echo("No selection recorded.")


@app.command("select-column-all")
def select_column_all_command(
    raw_dir: Path = typer.Option(
        Path("data/assay_tables"),
        "--raw-dir",
        help="Directory containing raw per-assay parquet tables.",
    ),
    aggregated_dir: Path = typer.Option(
        Path("outputs/aggregated"),
        "--aggregated-dir",
        help="Directory containing aggregated per-assay parquet tables.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Metadata SQLite DB containing assays and selections.",
    ),
    start_aid: Optional[int] = typer.Option(
        None,
        "--start-aid",
        help="Optional minimum AID; skip anything below this.",
    ),
) -> None:
    """Iteratively run select-column for all assays missing a selection."""
    import sqlite3
    import time
    from rich.console import Console

    if not metadata_db.exists():
        raise typer.BadParameter(f"Metadata DB not found: {metadata_db}")

    conn = sqlite3.connect(str(metadata_db))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT aid
            FROM assay_metadata
            WHERE (selected_column IS NULL OR selected_column = '')
            ORDER BY aid
            """
        )
        aids = [int(row["aid"]) for row in cur]
    finally:
        conn.close()

    if start_aid is not None:
        aids = [aid for aid in aids if aid >= start_aid]

    if not aids:
        typer.echo("No assays without selected_column found.")
        return

    typer.echo(f"Found {len(aids)} assays without selected_column.")
    typer.echo("You can stop at any time with Ctrl+C; already processed assays will be saved.\n")

    console = Console()

    def _fmt_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    start_time = time.perf_counter()
    total = len(aids)

    for idx, aid in enumerate(aids, start=1):
        elapsed = time.perf_counter() - start_time
        avg = elapsed / (idx - 1) if idx > 1 else 0.0
        remaining = total - idx + 1
        eta = avg * remaining if idx > 1 else 0.0

        # Print progress on its own line, separate from the per-assay table.
        console.print(
            f"[bold cyan]{idx}/{total}[/] AID {aid}  "
            f"[dim](elapsed {_fmt_hms(elapsed)}, ETA {_fmt_hms(eta)})[/dim]"
        )

        raw_path = raw_dir / f"aid_{aid}.parquet"
        agg_path = aggregated_dir / f"aid_{aid}_cid_agg.parquet"
        try:
            result = interactive_select_for_aid(
                aid=aid,
                aggregated_parquet=agg_path,
                raw_parquet=raw_path,
                metadata_db=metadata_db,
            )
        except KeyboardInterrupt:
            console.print("\n[red]Stopping selection loop on user interrupt.[/red]")
            break
        if result == "__INELIGIBLE__":
            console.print("Marked as ineligible.")
        elif result:
            console.print(f"Selection recorded for AID {aid}: {result}")
        else:
            console.print("No selection recorded for this AID (skipped).")


@app.command("compute-rscores")
def compute_rscores_command(
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="SQLite metadata database containing selected columns and stats.",
    ),
    aggregated_dir: Path = typer.Option(
        Path("outputs/aggregated"),
        "--aggregated-dir",
        help="Directory containing aggregated per-assay parquet tables.",
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
        Path("outputs/aggregated"),
        "--aggregated-dir",
        help="Directory containing aggregated per-assay parquet tables.",
    ),
) -> None:
    """Compute stats for already-selected columns and update metadata DB."""

    updated = 0
    for aid, selected_column in iter_selected_for_stats(metadata_db):
        agg_path = aggregated_dir / f"aid_{aid}_cid_agg.parquet"
        if not agg_path.exists():
            typer.echo(f"[skip] Aggregated parquet missing for AID {aid}: {agg_path}")
            continue
        row_count, stats = _compute_candidate_stats(aggregated_parquet=agg_path)
        match = next((c for c in stats if c.column_name == selected_column), None)
        if match is None:
            typer.echo(
                f"[warn] Selected column '{selected_column}' not found in candidates for AID {aid}."
            )
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

    typer.echo(f"Updated stats for {updated} assays in {metadata_db}")


@app.command("import-metadata-csv")
def import_metadata_csv_command(
    csv_path: Path = typer.Argument(
        Path("outputs/assay_metadata.csv"),
        help="Existing metadata CSV to import (one-time migration).",
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
        help="Optional single AID; omit to annotate all eligible assays.",
    ),
    metadata_db: Path = typer.Option(
        Path("outputs/assay_metadata.sqlite"),
        "--metadata-db",
        help="Metadata SQLite DB containing assays and selections.",
    ),
    model_provider: str = typer.Option(
        "xai",
        "--model-provider",
        help="LLM provider to use for annotation (e.g., xai, openai, vertexai).",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Optional specific model name (defaults per provider).",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        help="Optional reasoning_effort parameter for supported models.",
    ),
    max_hz: float = typer.Option(
        5.0,
        "--max-hz",
        help="Maximum PUG-REST request rate (Hz).",
    ),
    out_dir: Path = typer.Option(
        Path("outputs/llm_annotations"),
        "--out-dir",
        help="Directory to write JSON files with LLM annotations and rationales.",
    ),
) -> None:
    """Annotate target_type/bioactivity_type with an LLM based on assay descriptions."""
    # Load XAI/OpenAI/etc. keys from the project-local .env so the LLM factory can see them.
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    annotate_metadata_llm(
        metadata_db=metadata_db,
        aid=aid,
        model_provider=model_provider,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        max_hz=max_hz,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    app()

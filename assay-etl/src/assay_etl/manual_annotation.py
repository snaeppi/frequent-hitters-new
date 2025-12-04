"""Interactive (manual) assay metadata annotation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .assay_parser import AssayDescriptionRecord, parse_assay_description
from .metadata import AssayMetadata, _format_from_labels
from .meta_db import upsert_static_metadata

console = Console()

TARGET_CHOICES = {
    "1": "target-based",
    "2": "cell-based",
    "3": "other",
}

BIOACTIVITY_CHOICES = {
    "1": "specific bioactivity",
    "2": "nonspecific bioactivity",
    "3": "other",
}


@dataclass(slots=True)
class AnnotationContext:
    aid: int
    selected_column: str | None
    target_type: str | None
    bioactivity_type: str | None


def _fetch_assay_description_xml(
    *, aid: int, client: httpx.Client
) -> str:
    """Fetch assay description XML from PUG-REST."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/XML"
    resp = client.get(url, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def _format_record_text(record: AssayDescriptionRecord) -> Text:
    """Format description/protocol/comment into a single Text block for paging."""
    parts: list[str] = []
    if record.description:
        parts.append("Description:\n" + "\n\n".join(record.description))
    if record.protocol:
        parts.append("Protocol:\n" + "\n\n".join(record.protocol))
    if record.comment:
        parts.append("Comment:\n" + "\n\n".join(record.comment))
    if not parts:
        parts.append("No description or protocol text found for this assay.")
    return Text("\n\n".join(parts))


def _prompt_choice(
    *,
    label: str,
    choices: dict[str, str],
    existing: str | None,
) -> str | None:
    """Prompt for a choice, returning None when the user skips the assay."""
    options = "  ".join(f"[{key}] {value}" for key, value in choices.items())
    default_hint = f" (Enter to keep '{existing}')" if existing else ""
    prompt = f"{label}: {options}{default_hint}  [s=skip] "
    while True:
        raw = console.input(prompt).strip().lower()
        if raw in {"s", "skip"}:
            return None
        if not raw:
            if existing:
                return existing
            console.print("Please pick an option or 's' to skip.")
            continue
        choice = choices.get(raw)
        if choice:
            return choice
        console.print("Invalid choice. Use 1/2/3 or 's' to skip.")


def _prompt_annotation(
    *,
    existing_target: str | None,
    existing_bioactivity: str | None,
) -> tuple[str, str] | None:
    """Collect target_type and bioactivity_type selections from the user."""
    target = _prompt_choice(label="Target Type", choices=TARGET_CHOICES, existing=existing_target)
    if target is None:
        return None
    bio = _prompt_choice(
        label="Bioactivity Type",
        choices=BIOACTIVITY_CHOICES,
        existing=existing_bioactivity,
    )
    if bio is None:
        return None
    return target, bio


def _render_header(record: AssayDescriptionRecord, ctx: AnnotationContext, index: int, total: int) -> None:
    """Print a compact header with name, URL, and selection context."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{record.aid}"
    header = Text()
    header.append(f"{index}/{total}  AID {record.aid}", style="bold")
    if record.name:
        header.append(f" • {record.name}")
    header.append("\n")
    if record.source_name:
        header.append(f"Source: {record.source_name}\n", style="dim")
    if ctx.selected_column:
        header.append(f"Selected column: {ctx.selected_column}\n", style="dim")
    header.append("PubChem page: ")
    header.append(url, style=f"link {url}")
    console.print(Panel(header, title="Assay", expand=False))


def _row_to_context(row: sqlite3.Row) -> AnnotationContext:
    return AnnotationContext(
        aid=int(row["aid"]),
        selected_column=row["selected_column"],
        target_type=row["target_type"],
        bioactivity_type=row["bioactivity_type"],
    )


def load_annotation_context(
    *,
    metadata_db: Path,
    aid: int,
    include_annotated: bool = False,
    verbose: bool = True,
) -> AnnotationContext | None:
    """Load a single assay context, optionally skipping already annotated rows."""
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    conn = sqlite3.connect(str(metadata_db))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT aid, selected_column, target_type, bioactivity_type
            FROM assay_metadata
            WHERE aid = ?
            """,
            (aid,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        if verbose:
            console.print(f"[yellow]Skipping AID {aid}: not found in metadata DB.[/]")
        return None
    selected = row["selected_column"]
    if selected is None or selected == "" or selected == "__INELIGIBLE__":
        if verbose:
            console.print(f"[yellow]Skipping AID {aid}: no valid selected_column.[/]")
        return None
    if (
        not include_annotated
        and row["target_type"]
        and row["bioactivity_type"]
    ):
        if verbose:
            console.print(f"[green]Skipping AID {aid}: already annotated.[/]")
        return None
    return _row_to_context(row)


def iter_annotation_contexts(
    *,
    metadata_db: Path,
    aids: Optional[Sequence[int]] = None,
    start_aid: Optional[int] = None,
    include_annotated: bool = False,
) -> list[AnnotationContext]:
    """Collect contexts eligible for manual annotation."""
    if aids is not None:
        filtered = []
        for aid in sorted(set(aids)):
            if start_aid is not None and aid < start_aid:
                continue
            ctx = load_annotation_context(
                metadata_db=metadata_db,
                aid=aid,
                include_annotated=include_annotated,
                verbose=include_annotated,
            )
            if ctx is not None:
                filtered.append(ctx)
        return filtered

    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    conn = sqlite3.connect(str(metadata_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT aid, selected_column, target_type, bioactivity_type
            FROM assay_metadata
            WHERE selected_column IS NOT NULL
              AND selected_column != ''
              AND selected_column != '__INELIGIBLE__'
            ORDER BY aid
            """
        )
        contexts = []
        for row in cur:
            aid_val = int(row["aid"])
            if start_aid is not None and aid_val < start_aid:
                continue
            if (
                not include_annotated
                and row["target_type"]
                and row["bioactivity_type"]
            ):
                continue
            contexts.append(_row_to_context(row))
    finally:
        conn.close()

    return contexts


def annotate_metadata_manual(
    *,
    metadata_db: Path,
    context: AnnotationContext,
    client: httpx.Client,
    index: int,
    total: int,
) -> bool:
    """Interactively annotate target_type / bioactivity_type for a single assay."""
    try:
        xml_payload = _fetch_assay_description_xml(aid=context.aid, client=client)
        record = parse_assay_description(xml_payload)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to fetch/parse assay {context.aid}: {exc}[/]")
        return False

    _render_header(record, context, index, total)
    console.print("[dim]Press 'q' in the pager to return to the prompt.[/dim]")
    with console.pager(styles=True):
        console.print(_format_record_text(record))

    annotation = _prompt_annotation(
        existing_target=context.target_type,
        existing_bioactivity=context.bioactivity_type,
    )
    if annotation is None:
        console.print(f"[yellow]Skipped AID {context.aid} (left unchanged).[/]")
        return False

    target_type, bioactivity_type = annotation
    assay_format = _format_from_labels(
        target_type=target_type,
        bioactivity_type=bioactivity_type,
        has_annotation=True,
    )
    upsert_static_metadata(
        metadata_db,
        [
            AssayMetadata(
                aid=context.aid,
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
        ],
    )
    console.print(
        f"[green]Updated AID {context.aid}[/green]: "
        f"target_type='{target_type}', bioactivity_type='{bioactivity_type}', "
        f"assay_format='{assay_format or 'none'}'."
    )
    return True

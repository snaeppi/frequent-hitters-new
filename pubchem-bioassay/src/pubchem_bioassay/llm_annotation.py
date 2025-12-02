"""LLM-powered assay metadata annotation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import httpx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .llm import factory as llm_factory

from .assay_parser import AssayDescriptionRecord, parse_assay_description
from .meta_db import upsert_static_metadata
from .metadata import AssayMetadata, _format_from_labels
from .utils import RateLimiter


TARGET_DEFINITIONS = """Target Type definitions:
- target-based assay: readouts from purified proteins or peptides (biochemical preparations).
- cell-based assay: readouts from intact cells.
- other: tissue-based or organism-based assays not covered above."""

BIOACTIVITY_DEFINITIONS = """Bioactivity Type definitions:
- specific bioactivity: measures a specific biological property such as enzyme activity; excludes cytotoxicity.
- nonspecific bioactivity: measures cell growth, viability, cytotoxicity, or other nonspecific effects.
- other: physicochemical processes, DNA/RNA binding, or measurements outside biological activity."""

SYSTEM_PROMPT = f"""You are an expert bioassay curator.
Your task is to assign Target Type and Bioactivity Type strictly according to the definitions below.
{TARGET_DEFINITIONS}
{BIOACTIVITY_DEFINITIONS}
Base your decision solely on the provided assay description and protocol. When uncertain, choose "other". Keep the rationale concise (one or two sentences).
Respond using the requested structured schema."""

HUMAN_TEMPLATE = """
AID: {aid}
Name: {assay_name}
Source: {source_name}

Description:
{description_text}

Protocol:
{protocol_text}
"""


class MetadataAnnotation(BaseModel):
    target_type: Literal["target-based", "cell-based", "other"] = Field(
        description="Target type classification per definitions."
    )
    bioactivity_type: Literal["specific bioactivity", "nonspecific bioactivity", "other"] = Field(
        description="Bioactivity type classification per definitions."
    )
    rationale: str = Field(
        description="Short explanation citing evidence from the description/protocol.",
    )


def _fetch_assay_description_xml(
    *,
    aid: int,
    client: httpx.Client,
    limiter: RateLimiter,
) -> str:
    """Fetch assay description XML from PUG-REST."""
    limiter.wait()
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/XML"
    resp = client.get(url, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def _build_human_message(record: AssayDescriptionRecord) -> str:
    desc_text = "\n\n".join(record.description) if record.description else "n/a"
    proto_text = "\n\n".join(record.protocol) if record.protocol else "n/a"
    return HUMAN_TEMPLATE.format(
        aid=record.aid,
        assay_name=record.name or "Unknown",
        source_name=record.source_name or "Unknown",
        description_text=desc_text,
        protocol_text=proto_text,
    )


def annotate_metadata_llm(
    *,
    metadata_db: Path,
    aid: Optional[int] = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    max_hz: float = 5.0,
    out_dir: Path = Path("outputs/llm_annotations"),
) -> None:
    """Annotate assay metadata using an LLM and store results in SQLite + JSON files.

    - If `aid` is provided, only that assay is annotated.
    - Otherwise, all assays with:
        selected_column set (not empty and not '__INELIGIBLE__')
        AND both target_type and bioactivity_type missing
      will be annotated.
    """
    import sqlite3

    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM and LangChain agent using the shared factory and Pydantic schema,
    # matching the original pipeline implementation.
    llm, resolved_model_name = llm_factory.create_llm(
        model_provider=model_provider,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    agent = create_agent(
        model=llm,
        tools=[],
        response_format=MetadataAnnotation,
    )

    conn = sqlite3.connect(str(metadata_db))
    try:
        conn.row_factory = sqlite3.Row
        if aid is not None:
            rows = conn.execute(
                """
                SELECT aid
                FROM assay_metadata
                WHERE aid = ?
                  AND selected_column IS NOT NULL
                  AND selected_column != ''
                  AND selected_column != '__INELIGIBLE__'
                  AND (target_type IS NULL OR target_type = '')
                  AND (bioactivity_type IS NULL OR bioactivity_type = '')
                """,
                (aid,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT aid
                FROM assay_metadata
                WHERE selected_column IS NOT NULL
                  AND selected_column != ''
                  AND selected_column != '__INELIGIBLE__'
                  AND (target_type IS NULL OR target_type = '')
                  AND (bioactivity_type IS NULL OR bioactivity_type = '')
                ORDER BY aid
                """
            ).fetchall()
        aids = [int(row["aid"]) for row in rows]
    finally:
        conn.close()

    if not aids:
        print("No assays require LLM-based annotation.")
        return

    http_client = httpx.Client(timeout=60.0)
    limiter = RateLimiter(max_per_second=max_hz)

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Annotating assays", total=len(aids))
            for idx, target_aid in enumerate(aids, start=1):
                progress.update(task_id, description=f"Annotating AID {target_aid}")
                xml_payload = _fetch_assay_description_xml(
                    aid=target_aid, client=http_client, limiter=limiter
                )
                record = parse_assay_description(xml_payload)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_build_human_message(record)),
                ]
                agent_state = agent.invoke({"messages": messages})
                response = agent_state.get("structured_response")
                if response is None:
                    raise RuntimeError(
                        f"Metadata annotation returned no structured_response for AID {target_aid}."
                    )
                annotation: MetadataAnnotation = response

                # Compute assay_format using the same mapping as Hit Dexter.
                assay_format = _format_from_labels(
                    target_type=annotation.target_type,
                    bioactivity_type=annotation.bioactivity_type,
                    has_annotation=True,
                )

                upsert_static_metadata(
                    metadata_db,
                    [
                        AssayMetadata(
                            aid=target_aid,
                            target_type=annotation.target_type,
                            bioactivity_type=annotation.bioactivity_type,
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

                # Save annotation + rationale for manual inspection.
                payload = {
                    "aid": target_aid,
                    "model_provider": model_provider or "xai",
                    "model_name": resolved_model_name,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "annotation": annotation.model_dump(),
                    "description": record.as_json_dict(),
                }
                out_path = out_dir / f"aid_{target_aid}_annotation.json"
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                progress.advance(task_id)
    finally:
        http_client.close()

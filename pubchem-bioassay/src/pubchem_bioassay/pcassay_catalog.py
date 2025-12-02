"""Parser for the provided pcassay_result.txt catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class PcassayRecord:
    aid: int
    title: str | None = None
    source: str | None = None
    protein_target: str | None = None
    bioactivity: str | None = None
    extra_lines: tuple[str, ...] = ()


AID_RE = re.compile(r"^AID:\s*(\d+)\s*$")
TITLE_RE = re.compile(r"^\s*\d+\.\s+(.*)$")


def load_pcassay_results(path: Path) -> Dict[int, PcassayRecord]:
    """Return a mapping of AID -> record parsed from the text export."""
    if not path.exists():
        raise FileNotFoundError(f"pcassay_result file not found: {path}")

    records: dict[int, PcassayRecord] = {}
    current_title: str | None = None
    current_source: str | None = None
    current_target: str | None = None
    current_bioactivity: str | None = None
    extra_lines: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            if match := TITLE_RE.match(stripped):
                current_title = match.group(1).strip()
                current_source = None
                current_target = None
                current_bioactivity = None
                extra_lines = []
                continue

            if stripped.startswith("Source:"):
                current_source = stripped.split(":", 1)[1].strip() or None
                continue

            if stripped.startswith("Protein Target:"):
                current_target = stripped.split(":", 1)[1].strip() or None
                continue

            if stripped.startswith("Substance BioActivity:"):
                current_bioactivity = stripped.split(":", 1)[1].strip() or None
                continue

            if match := AID_RE.match(stripped):
                aid = int(match.group(1))
                records[aid] = PcassayRecord(
                    aid=aid,
                    title=current_title,
                    source=current_source,
                    protein_target=current_target,
                    bioactivity=current_bioactivity,
                    extra_lines=tuple(extra_lines),
                )
                current_title = None
                current_source = None
                current_target = None
                current_bioactivity = None
                extra_lines = []
                continue

            extra_lines.append(stripped)

    return records


def iter_aids_from_pcassay(path: Path) -> Iterable[int]:
    """Yield AIDs from a pcassay_result.txt file."""
    records = load_pcassay_results(path)
    return records.keys()



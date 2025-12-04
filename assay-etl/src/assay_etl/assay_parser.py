"""Utilities for parsing PUG-REST assay description XML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from lxml import etree

PC_NS = {"pc": "http://www.ncbi.nlm.nih.gov"}


def _text_list(nodes: List[etree._Element]) -> List[str]:
    return [node.text.strip() for node in nodes if node.text and node.text.strip()]


@dataclass(slots=True)
class AssayDescriptionRecord:
    aid: int
    name: Optional[str]
    source_name: Optional[str]
    description: List[str]
    protocol: List[str]
    comment: List[str]
    activity_outcome_method: Optional[str]
    project_category: Optional[str]
    raw_xml: str

    def as_json_dict(self) -> Dict[str, Any]:
        return {
            "aid": self.aid,
            "name": self.name,
            "source_name": self.source_name,
            "description": self.description,
            "protocol": self.protocol,
            "comment": self.comment,
            "activity_outcome_method": self.activity_outcome_method,
            "project_category": self.project_category,
        }


def parse_assay_description(xml_payload: str) -> AssayDescriptionRecord:
    """Parse the PUG-REST assay description XML into a structured record."""
    root = etree.fromstring(xml_payload.encode("utf-8"))
    desc_node = root.find(".//pc:PC-AssayDescription", namespaces=PC_NS)
    if desc_node is None:
        raise ValueError("Unable to find PC-AssayDescription node")

    aid = int(desc_node.findtext(".//pc:PC-ID_id", namespaces=PC_NS, default="0"))
    name = desc_node.findtext("pc:PC-AssayDescription_name", namespaces=PC_NS)
    source_name = desc_node.findtext(
        ".//pc:PC-DBTracking_name",
        namespaces=PC_NS,
    )

    description_sections = _text_list(
        desc_node.findall(
            ".//pc:PC-AssayDescription_description/pc:PC-AssayDescription_description_E",
            namespaces=PC_NS,
        )
    )
    protocol_sections = _text_list(
        desc_node.findall(
            ".//pc:PC-AssayDescription_protocol/pc:PC-AssayDescription_protocol_E",
            namespaces=PC_NS,
        )
    )
    comment_sections = _text_list(
        desc_node.findall(
            ".//pc:PC-AssayDescription_comment/pc:PC-AssayDescription_comment_E",
            namespaces=PC_NS,
        )
    )

    activity_node = desc_node.find(
        "pc:PC-AssayDescription_activity-outcome-method", namespaces=PC_NS
    )
    activity_outcome_method = (
        activity_node.get("value") if activity_node is not None else None
    )

    project_node = desc_node.find(
        "pc:PC-AssayDescription_project-category", namespaces=PC_NS
    )
    project_category = project_node.get("value") if project_node is not None else None

    return AssayDescriptionRecord(
        aid=aid,
        name=name,
        source_name=source_name,
        description=description_sections,
        protocol=protocol_sections,
        comment=comment_sections,
        activity_outcome_method=activity_outcome_method,
        project_category=project_category,
        raw_xml=xml_payload,
    )


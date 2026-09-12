"""Loading and validating videos_metadata.json, and resolving provisional
grip labels for a given (video, timestamp) pair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.config import VALID_GRIPS, VALID_SHOT_CONTEXTS, VALID_TYPES


class MetadataError(ValueError):
    pass


@dataclass
class Section:
    start: int
    end: int
    grip: str
    shot_context: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class VideoEntry:
    id: str
    youtube_url: str
    type: str
    grip: Optional[str] = None
    shot_context: Optional[str] = None
    sections: list[Section] = field(default_factory=list)
    notes: Optional[str] = None

    def provisional_label(self, timestamp: int) -> tuple[Optional[str], Optional[str], bool]:
        """Return (grip, shot_context, in_scope) for a frame at `timestamp` seconds.

        in_scope is False only for a sectioned video whose timestamp falls
        outside every section (that frame must be skipped entirely).
        """
        if self.type == "single_grip":
            return self.grip, self.shot_context, True
        if self.type == "sectioned":
            for sec in self.sections:
                if sec.start <= timestamp < sec.end:
                    return sec.grip, sec.shot_context, True
            return None, None, False
        # unlabeled
        return None, None, True


def _validate_grip(grip: Optional[str], where: str) -> None:
    if grip is not None and grip not in VALID_GRIPS:
        raise MetadataError(f"{where}: invalid grip {grip!r}, must be one of {VALID_GRIPS}")


def _validate_shot_context(shot_context: Optional[str], where: str) -> None:
    if shot_context is not None and shot_context not in VALID_SHOT_CONTEXTS:
        raise MetadataError(
            f"{where}: invalid shot_context {shot_context!r}, must be one of {VALID_SHOT_CONTEXTS}"
        )


def load_metadata(path: Path) -> list[VideoEntry]:
    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise MetadataError(f"{path}: top-level JSON must be a list of video entries")

    entries: list[VideoEntry] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(raw):
        where = f"videos_metadata.json entry {i} (id={item.get('id')!r})"

        for required in ("id", "youtube_url", "type"):
            if required not in item:
                raise MetadataError(f"{where}: missing required field {required!r}")

        vid_id = item["id"]
        if vid_id in seen_ids:
            raise MetadataError(f"{where}: duplicate id {vid_id!r}")
        seen_ids.add(vid_id)

        vtype = item["type"]
        if vtype not in VALID_TYPES:
            raise MetadataError(f"{where}: invalid type {vtype!r}, must be one of {VALID_TYPES}")

        grip = item.get("grip")
        shot_context = item.get("shot_context")
        notes = item.get("notes")
        sections: list[Section] = []

        if vtype == "single_grip":
            if not grip:
                raise MetadataError(f"{where}: type=single_grip requires a 'grip' field")
            _validate_grip(grip, where)
            _validate_shot_context(shot_context, where)

        elif vtype == "sectioned":
            raw_sections = item.get("sections")
            if not raw_sections:
                raise MetadataError(f"{where}: type=sectioned requires a non-empty 'sections' list")
            for j, s in enumerate(raw_sections):
                swhere = f"{where} section {j}"
                for required in ("start", "end", "grip"):
                    if required not in s:
                        raise MetadataError(f"{swhere}: missing required field {required!r}")
                if s["end"] <= s["start"]:
                    raise MetadataError(f"{swhere}: end must be > start")
                _validate_grip(s["grip"], swhere)
                _validate_shot_context(s.get("shot_context"), swhere)
                sections.append(
                    Section(
                        start=int(s["start"]),
                        end=int(s["end"]),
                        grip=s["grip"],
                        shot_context=s.get("shot_context"),
                        notes=s.get("notes"),
                    )
                )

        else:  # unlabeled
            if grip or item.get("sections"):
                raise MetadataError(
                    f"{where}: type=unlabeled must not have 'grip' or 'sections' fields"
                )

        entries.append(
            VideoEntry(
                id=vid_id,
                youtube_url=item["youtube_url"],
                type=vtype,
                grip=grip,
                shot_context=shot_context,
                sections=sections,
                notes=notes,
            )
        )

    return entries

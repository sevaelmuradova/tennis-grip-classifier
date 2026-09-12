"""Tracks which videos have been fully processed (frames extracted, filtered,
deduped, LLM-verified, and written to candidates/) so re-running the pipeline
doesn't redo that work. Keyed separately from the manifest because a video
can legitimately produce zero candidate frames."""

from __future__ import annotations

import json
from pathlib import Path


def load_processed(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def mark_processed(state_path: Path, video_id: str, summary: dict) -> None:
    state = load_processed(state_path)
    state[video_id] = summary
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

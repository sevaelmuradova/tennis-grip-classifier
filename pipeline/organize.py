"""Writing candidate frames to disk and appending rows to manifest.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from pipeline.config import OUTPUT_LABELS

MANIFEST_FIELDS = [
    "filename",
    "source_id",
    "source_youtube_url",
    "timestamp_seconds",
    "hand_confidence",
    "provisional_label",
    "provisional_shot_context",
    "llm_label",
    "llm_confidence",
    "llm_reasoning",
    "final_label",
    "needs_review",
]


def ensure_candidate_dirs(candidates_dir: Path) -> None:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for label in OUTPUT_LABELS:
        (candidates_dir / label).mkdir(exist_ok=True)


class ManifestWriter:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        is_new = not manifest_path.exists()
        self._fh = open(manifest_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=MANIFEST_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_candidate_frame(
    candidates_dir: Path,
    jpg_bytes: bytes,
    final_label: str,
    source_id: str,
    timestamp_seconds: int,
    hand_confidence: float,
) -> str:
    """Write the frame's JPEG bytes into candidates/{final_label}/ and return
    the filename (relative to that subfolder)."""
    filename = f"{source_id}_{timestamp_seconds}_{hand_confidence:.2f}.jpg"
    out_path = candidates_dir / final_label / filename
    with open(out_path, "wb") as f:
        f.write(jpg_bytes)
    return filename


def manifest_row(
    filename: str,
    source_id: str,
    source_youtube_url: str,
    timestamp_seconds: int,
    hand_confidence: float,
    provisional_label: Optional[str],
    provisional_shot_context: Optional[str],
    llm_label: str,
    llm_confidence: Optional[str],
    llm_reasoning: str,
    final_label: str,
    needs_review: bool,
) -> dict:
    return {
        "filename": filename,
        "source_id": source_id,
        "source_youtube_url": source_youtube_url,
        "timestamp_seconds": timestamp_seconds,
        "hand_confidence": f"{hand_confidence:.4f}",
        "provisional_label": provisional_label or "",
        "provisional_shot_context": provisional_shot_context or "",
        "llm_label": llm_label,
        "llm_confidence": llm_confidence or "",
        "llm_reasoning": llm_reasoning,
        "final_label": final_label,
        "needs_review": needs_review,
    }

"""Perceptual-hash based de-duplication of consecutive kept frames."""

from __future__ import annotations

import io
from typing import NamedTuple

import imagehash
from PIL import Image

from pipeline.config import DEDUP_SIMILARITY_THRESHOLD, PHASH_SIZE


class KeptFrame(NamedTuple):
    timestamp: int
    jpg_bytes: bytes
    grip: str | None
    shot_context: str | None
    confidence: float
    bbox_area_ratio: float


def _phash(jpg_bytes: bytes) -> imagehash.ImageHash:
    with Image.open(io.BytesIO(jpg_bytes)) as img:
        return imagehash.phash(img, hash_size=PHASH_SIZE)


def dedup_frames(
    frames: list[KeptFrame],
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> list[KeptFrame]:
    """Given frames sorted by timestamp, drop any frame that is more than
    `similarity_threshold` visually similar to the immediately preceding
    *kept* frame (within the same video)."""
    if not frames:
        return []

    total_bits = PHASH_SIZE * PHASH_SIZE

    kept: list[KeptFrame] = [frames[0]]
    last_hash = _phash(frames[0].jpg_bytes)

    for frame in frames[1:]:
        h = _phash(frame.jpg_bytes)
        distance = h - last_hash
        similarity = 1 - (distance / total_bits)
        if similarity > similarity_threshold:
            continue  # too similar to the last kept frame; drop it
        kept.append(frame)
        last_hash = h

    return kept

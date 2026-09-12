#!/usr/bin/env python3
"""Tennis grip classification data pipeline.

Downloads tutorial videos, extracts frames, filters for a single confidently
detected hand via MediaPipe, provisionally labels frames from per-video
metadata, verifies/labels with the Gemini API, and organizes the results
into ./candidates/ for quick human review.

Usage:
    python main.py
    python main.py --metadata videos_metadata.json --limit 2
    python main.py --hand-confidence 0.8 --bbox-ratio 0.2 --dedup-threshold 0.85

See README.md for full documentation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

from pipeline import config
from pipeline.dedup import KeptFrame, dedup_frames
from pipeline.downloader import download_video
from pipeline.frame_extractor import extract_frames
from pipeline.hand_filter import HandFilter
from pipeline.labeling import assign_final_label
from pipeline.llm_client import LLMClient
from pipeline.metadata import MetadataError, load_metadata
from pipeline.organize import ensure_candidate_dirs, manifest_row, save_candidate_frame, ManifestWriter
from pipeline.state import load_processed, mark_processed

load_dotenv()  # picks up GEMINI_API_KEY from a .env file in the project root, if present

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pipeline.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", default=str(config.METADATA_PATH), help="path to videos_metadata.json")
    p.add_argument("--raw-videos-dir", default=str(config.RAW_VIDEOS_DIR))
    p.add_argument("--candidates-dir", default=str(config.CANDIDATES_DIR))
    p.add_argument("--hand-confidence", type=float, default=config.HAND_MIN_DETECTION_CONFIDENCE,
                    help="minimum MediaPipe hand detection confidence (default: %(default)s)")
    p.add_argument("--bbox-ratio", type=float, default=config.HAND_MIN_BBOX_AREA_RATIO,
                    help="minimum hand bounding-box area as a fraction of frame area (default: %(default)s)")
    p.add_argument("--dedup-threshold", type=float, default=config.DEDUP_SIMILARITY_THRESHOLD,
                    help="perceptual-hash similarity above which consecutive frames are deduped (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N videos (for testing)")
    p.add_argument("--skip-llm", action="store_true", help="skip LLM verification (for testing extraction/filtering only)")
    return p.parse_args()


def describe_entry(entry) -> str:
    if entry.type == "sectioned":
        return f"{entry.id}, sectioned, {len(entry.sections)} sections"
    return f"{entry.id}, {entry.type}"


def process_video(entry, args, hand_filter: HandFilter, llm_client, manifest: ManifestWriter, video_num: int, total: int) -> dict:
    raw_dir = Path(args.raw_videos_dir)
    candidates_dir = Path(args.candidates_dir)

    prefix = f"Video {video_num}/{total} ({describe_entry(entry)}):"

    video_path = download_video(entry, raw_dir)
    if video_path is None:
        logger.info("%s download FAILED, skipping", prefix)
        return {"status": "download_failed"}

    extracted = 0
    in_scope = 0
    kept: list[KeptFrame] = []

    for timestamp, frame in extract_frames(video_path):
        extracted += 1
        grip, shot_context, scope_ok = entry.provisional_label(timestamp)
        if not scope_ok:
            continue
        in_scope += 1

        detection = hand_filter.check(frame)
        if detection is None:
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ok:
            continue

        kept.append(
            KeptFrame(
                timestamp=timestamp,
                jpg_bytes=buf.tobytes(),
                grip=grip,
                shot_context=shot_context,
                confidence=detection.confidence,
                bbox_area_ratio=detection.bbox_area_ratio,
            )
        )

    passed_mediapipe = len(kept)
    deduped = dedup_frames(kept, similarity_threshold=args.dedup_threshold)
    kept_after_dedup = len(deduped)

    scope_note = f"{in_scope} in-section, " if entry.type == "sectioned" else ""
    logger.info(
        "%s downloaded, extracted %d frames, %s%d passed MediaPipe, %d kept after dedup, verifying with LLM...",
        prefix, extracted, scope_note, passed_mediapipe, kept_after_dedup,
    )

    final_label_counts: dict[str, int] = {}

    for frame in deduped:
        if args.skip_llm:
            continue

        if frame.grip is not None:
            llm_result = llm_client.verify_with_provisional(frame.jpg_bytes, frame.grip)
        else:
            llm_result = llm_client.label_without_provisional(frame.jpg_bytes)

        final_label, needs_review, llm_label = assign_final_label(frame.grip, llm_result)

        filename = save_candidate_frame(
            candidates_dir, frame.jpg_bytes, final_label, entry.id, frame.timestamp, frame.confidence
        )
        manifest.write_row(
            manifest_row(
                filename=filename,
                source_id=entry.id,
                source_youtube_url=entry.youtube_url,
                timestamp_seconds=frame.timestamp,
                hand_confidence=frame.confidence,
                provisional_label=frame.grip,
                provisional_shot_context=frame.shot_context,
                llm_label=llm_label,
                llm_confidence=llm_result.confidence,
                llm_reasoning=llm_result.reasoning,
                final_label=final_label,
                needs_review=needs_review,
            )
        )
        final_label_counts[final_label] = final_label_counts.get(final_label, 0) + 1

    logger.info("%s done. Final labels: %s", prefix, final_label_counts or "(none)")

    return {
        "status": "ok",
        "extracted": extracted,
        "in_scope": in_scope,
        "passed_mediapipe": passed_mediapipe,
        "kept_after_dedup": kept_after_dedup,
        "final_label_counts": final_label_counts,
    }


def main() -> int:
    args = parse_args()

    metadata_path = Path(args.metadata)
    candidates_dir = Path(args.candidates_dir)
    processed_path = candidates_dir / "processed_videos.json"

    try:
        entries = load_metadata(metadata_path)
    except MetadataError as e:
        logger.error("Invalid metadata: %s", e)
        return 1

    if args.limit is not None:
        entries = entries[: args.limit]

    ensure_candidate_dirs(candidates_dir)
    processed = load_processed(processed_path)

    llm_client = None if args.skip_llm else LLMClient()

    with HandFilter(
        min_detection_confidence=args.hand_confidence,
        min_bbox_area_ratio=args.bbox_ratio,
    ) as hand_filter, ManifestWriter(candidates_dir / "manifest.csv") as manifest:

        for i, entry in enumerate(entries, start=1):
            if entry.id in processed:
                logger.info("Video %d/%d (%s): already processed, skipping", i, len(entries), describe_entry(entry))
                continue

            result = process_video(entry, args, hand_filter, llm_client, manifest, i, len(entries))

            # Only record completion when the full pipeline (including LLM
            # labeling) ran -- a --skip-llm test run must not cause a later
            # real run to skip this video.
            if result["status"] == "ok" and not args.skip_llm:
                mark_processed(processed_path, entry.id, result)

    logger.info("Pipeline complete. See %s and %s", candidates_dir / "manifest.csv", candidates_dir / "conflicted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

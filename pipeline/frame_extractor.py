"""1-fps frame extraction via OpenCV, trimming the first/last N seconds."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from pipeline.config import EDGE_TRIM_SECONDS, FRAME_INTERVAL_SECONDS


def extract_frames(video_path: Path) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Yield (timestamp_seconds, frame_bgr) at FRAME_INTERVAL_SECONDS resolution,
    skipping the first and last EDGE_TRIM_SECONDS of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps else 0

        last_valid_ts = int(duration) - EDGE_TRIM_SECONDS
        if last_valid_ts <= EDGE_TRIM_SECONDS:
            return  # video too short to contain any in-scope frames

        for t in range(EDGE_TRIM_SECONDS, last_valid_ts + 1, FRAME_INTERVAL_SECONDS):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            yield t, frame
    finally:
        cap.release()

"""Idempotent video downloading via yt-dlp."""

from __future__ import annotations

import logging
from pathlib import Path

import yt_dlp

from pipeline.config import YDL_FORMAT
from pipeline.metadata import VideoEntry

logger = logging.getLogger("pipeline.downloader")


def download_video(entry: VideoEntry, raw_dir: Path) -> Path | None:
    """Download entry.youtube_url to raw_dir/{entry.id}.mp4 if not already present.

    Returns the path to the video file, or None if the download failed
    (video unavailable, age-restricted, region-blocked, etc.) -- the error
    is logged and the caller should skip this video and continue.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{entry.id}.mp4"

    if target.exists():
        logger.info("  [%s] already downloaded -> %s", entry.id, target)
        return target

    ydl_opts = {
        "format": YDL_FORMAT,
        "outtmpl": str(raw_dir / f"{entry.id}.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([entry.youtube_url])
    except yt_dlp.utils.DownloadError as e:
        logger.warning("  [%s] download FAILED (%s): %s", entry.id, entry.youtube_url, e)
        return None
    except Exception as e:  # noqa: BLE001 - any other yt-dlp/network failure
        logger.warning("  [%s] download FAILED unexpectedly (%s): %s", entry.id, entry.youtube_url, e)
        return None

    if not target.exists():
        # yt-dlp sometimes produces a different extension if merging failed;
        # look for any file with the right stem and normalize it.
        candidates = list(raw_dir.glob(f"{entry.id}.*"))
        if candidates:
            candidates[0].rename(target)

    if not target.exists():
        logger.warning("  [%s] download reported success but %s not found", entry.id, target)
        return None

    logger.info("  [%s] downloaded -> %s", entry.id, target)
    return target

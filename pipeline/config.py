"""Central constants and tunable thresholds for the grip-labeling pipeline.

All of these can be overridden via CLI flags on main.py (see main.py --help);
the values here are just the defaults.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "videos_metadata.json"
RAW_VIDEOS_DIR = PROJECT_ROOT / "raw_videos"
CANDIDATES_DIR = PROJECT_ROOT / "candidates"
MANIFEST_PATH = CANDIDATES_DIR / "manifest.csv"
PROCESSED_VIDEOS_PATH = CANDIDATES_DIR / "processed_videos.json"

VALID_GRIPS = ("continental", "eastern", "semi_western", "western")
VALID_SHOT_CONTEXTS = ("forehand", "backhand", "serve", "volley")
VALID_TYPES = ("single_grip", "sectioned", "unlabeled")

# Output buckets under candidates/
OUTPUT_LABELS = VALID_GRIPS + ("conflicted", "unclear")

# --- yt-dlp ---
YDL_FORMAT = "bestvideo[height<=720]+bestaudio/best[height<=720]"

# --- frame extraction ---
FRAME_INTERVAL_SECONDS = 1
EDGE_TRIM_SECONDS = 5  # skip first/last N seconds of each video

# --- MediaPipe Hands filtering ---
HAND_MIN_DETECTION_CONFIDENCE = 0.75
HAND_MIN_BBOX_AREA_RATIO = 0.15  # bounding box must cover >= 15% of frame area

# --- de-duplication (perceptual hashing) ---
DEDUP_SIMILARITY_THRESHOLD = 0.90  # frames more similar than this are considered dupes
PHASH_SIZE = 16  # hash_size for imagehash.phash; bigger = more sensitive to detail

# --- LLM verification (Gemini API, free tier) ---
# Check https://ai.google.dev/gemini-api/docs/models and .../rate-limits for
# the current recommended free-tier flash model and RPM cap -- both change
# over time and the values below may need bumping/updating.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_REQUESTS_PER_MINUTE = 10  # conservative default for the free tier
LLM_MAX_RETRIES = 1  # one retry after the first failure, then mark unclear
LLM_MAX_TOKENS = 300

JPEG_QUALITY = 90

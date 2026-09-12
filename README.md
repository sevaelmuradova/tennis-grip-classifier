# Tennis Grip Classification Pipeline

## What this project is

Tennis players often don't know which grip they're actually using, and the
grip you use changes what shots are natural, what spin you can produce, and
what injuries you're prone to. The long-term goal of this project is a tool
that can look at an image of someone's hand on a racket and tell them:
continental, eastern, semi-western, or western — the kind of instant
feedback a coach gives, available to anyone.

Getting there requires an image classifier, and a classifier needs a
labeled training set that doesn't exist anywhere off the shelf. That's what
this repository currently builds: a labeled dataset of tennis-grip frames,
pulled from real coaching footage on YouTube, provisionally labeled from
what each video is known to teach, and checked by an LLM so a human only
has to review the uncertain cases by hand instead of every frame. Training
the actual classifier on this dataset is a separate, later phase of the
project — this stage's job ends at handing over clean, pre-sorted candidate
images.

**Who it's for:** This tool is built for amateur and developing tennis players who don't have regular access to a coach's eye and for the coaches trying to reach more of them. Grip is one of the most fundamental technical elements in tennis, but it's also one of the hardest things to self-diagnose: most players can't see their own hand position clearly, and most self-training apps focus on full-swing mechanics rather than this specific, foundational detail.

The tool lets a player upload a short video or photo of their grip and get back an identification of which grip they're using (Continental, Eastern, Semi-Western, or Western), what that grip is typically used for, and whether it matches the shot they're working on. A coach can review any submission, correct the tool if it's wrong, and add their own notes — so the AI supports the coaching relationship instead of replacing it.

It's built specifically for players and coaches without access to expensive professional-grade analysis systems.

## This stage: the data pipeline

Downloads tutorial videos, extracts frames, filters for a single confidently
detected hand via MediaPipe, provisionally labels frames from per-video
metadata, verifies/labels with the Gemini API, and organizes the results
into `./candidates/` for quick human review.

## 1. Install dependencies

### System dependency: ffmpeg

`yt-dlp` shells out to `ffmpeg` to merge separately-downloaded video/audio
streams and to remux to `.mp4`. Install it first:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Verify it's installed and on your `PATH`:

```bash
ffmpeg -version
```

You should see a version banner (e.g. `ffmpeg version 6.x ...`). If you get
`command not found`, `yt-dlp` will still download videos in some cases but
will fail to merge separate video+audio streams into a single `.mp4` — the
pipeline's downloads will fail or produce audio-less files.

### Python dependencies

Requires Python 3.10+. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Gemini API key (free tier)

The LLM verification/labeling stage calls the Gemini API. Get a free key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no credit
card required).

A `.env` file already exists at the project root — open it in VS Code and
paste your key in:

```
GEMINI_API_KEY=your-key-here
```

`main.py` loads it automatically via `python-dotenv` (already in
`requirements.txt`). `.env` is gitignored, so the key never gets committed.
If you'd rather not use a `.env` file, exporting it in your shell works too
and takes precedence:

```bash
export GEMINI_API_KEY=...
```

The free tier is rate-limited (requests per minute), which is why
`main.py` paces LLM calls conservatively by default
(`GEMINI_MAX_REQUESTS_PER_MINUTE` in `pipeline/config.py`, currently 10).
Check [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
for your account's current limits and raise that constant if you have more
headroom, or lower it if you see 429/`RESOURCE_EXHAUSTED` errors. Also check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
for the current recommended free-tier flash model — `GEMINI_MODEL` in
`pipeline/config.py` may need updating if the model named there has been
superseded or retired.

## 2. Structure videos_metadata.json

A JSON list of video entries at the project root (`videos_metadata.json`).
Each entry needs an `id` (used as filename stem and in the manifest), a
`youtube_url`, and a `type` of `single_grip`, `sectioned`, or `unlabeled`:

- **`single_grip`** — the whole video shows one grip. Requires a top-level
  `grip` field; `shot_context` is optional.

  ```json
  {
    "id": "total_tennis_domination",
    "youtube_url": "https://youtu.be/defhZn5sdWQ",
    "type": "single_grip",
    "grip": "continental",
    "shot_context": "serve"
  }
  ```

- **`sectioned`** — different time ranges of the video show different
  grips. Requires a `sections` list; each section needs `start`/`end`
  (seconds) and `grip`, with optional `shot_context`. Frames outside every
  section are skipped entirely — they are never extracted for labeling.

  ```json
  {
    "id": "intuitive_tennis",
    "youtube_url": "https://youtu.be/WEMbmDurvNk",
    "type": "sectioned",
    "sections": [
      {
        "start": 257,
        "end": 355,
        "grip": "continental",
        "shot_context": "forehand"
      },
      {
        "start": 355,
        "end": 395,
        "grip": "eastern",
        "shot_context": "forehand"
      }
    ]
  }
  ```

- **`unlabeled`** — no `grip` or `sections` field at all. Every extracted
  frame goes straight to LLM primary labeling with no provisional label to
  check against.

  ```json
  {
    "id": "some_video",
    "youtube_url": "https://youtu.be/xxxxxxxxxxx",
    "type": "unlabeled"
  }
  ```

Valid `grip` values: `continental`, `eastern`, `semi_western`, `western`.
Valid `shot_context` values (optional): `forehand`, `backhand`, `serve`,
`volley`.

## 3. Run the pipeline

```bash
python main.py
```

This downloads any videos not already in `./raw_videos/`, extracts and
filters frames, calls the Gemini API, and writes results to
`./candidates/{grip}|conflicted|unclear/` plus `./candidates/manifest.csv`.

Useful flags:

```bash
python main.py --limit 1                     # process only the first video (smoke test)
python main.py --skip-llm                     # run download/extract/filter only, skip API calls
python main.py --hand-confidence 0.8          # tune MediaPipe confidence threshold
python main.py --bbox-ratio 0.20              # tune minimum hand bounding-box size
python main.py --dedup-threshold 0.85         # tune perceptual-hash dedup similarity
```

The pipeline is **idempotent**: re-running skips videos already downloaded
(`raw_videos/{id}.mp4` exists) and skips videos already fully processed
(tracked in `candidates/processed_videos.json`), so it's safe to re-run
after a crash, a metadata edit for a not-yet-processed video, or to add new
entries to `videos_metadata.json` without reprocessing everything.

To force a full reprocess of a single video, delete its entry from
`candidates/processed_videos.json` and remove its rows from
`candidates/manifest.csv` (its already-saved frame files in `candidates/`
will just be overwritten with the same filenames).

## 4. Review the output — start with `candidates/conflicted/`

`candidates/manifest.csv` has one row per candidate frame:

| column                                           | meaning                                                                                 |
| ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `filename`                                       | JPEG filename, also the path under `candidates/{final_label}/`                          |
| `source_id` / `source_youtube_url`               | which video the frame came from                                                         |
| `timestamp_seconds`                              | position in the source video                                                            |
| `hand_confidence`                                | MediaPipe detection confidence for the kept hand                                        |
| `provisional_label` / `provisional_shot_context` | from your metadata, if any                                                              |
| `llm_label`                                      | grip the LLM identified (or echoed back if it agreed)                                   |
| `llm_confidence` / `llm_reasoning`               | LLM's confidence and one-line rationale                                                 |
| `final_label`                                    | one of the four grips, `conflicted`, or `unclear`                                       |
| `needs_review`                                   | `True` when the LLM agreed with the provisional label but at only medium/low confidence |

**Recommended review order:**

1. **`candidates/conflicted/` first.** These are frames where your metadata
   said one grip but the LLM's vision read of the frame said another. This
   is the highest-value folder to review by hand — it's small, and each
   image is either a genuine metadata error (fix `videos_metadata.json` and
   re-run), a frame that's ambiguous/transitional, or an LLM vision miss.
   Sort/scan by `source_id` to catch systematic issues per video.
2. **`candidates/unclear/` next.** For `unlabeled`-type videos this is
   where the LLM couldn't confidently identify a grip; for labeled videos,
   this only happens when an LLM call failed twice in a row (rare). Spot
   check a handful — a large cluster here usually means the hand/racket
   isn't clearly visible and MediaPipe's bounding-box filter should
   probably be tightened.
3. **Rows with `needs_review=True` in the grip folders.** These already
   have a trustworthy-ish label (provisional label, LLM agreed) but only at
   medium/low LLM confidence — a quick pass here catches edge cases before
   they enter your training set.
4. **Everything else** (`needs_review=False` in a grip folder, i.e. LLM
   agreed at high confidence) is the highest-trust tier and generally
   doesn't need manual review before use, though a random spot-check is
   always a good idea before training on it.

## Project layout

```
videos_metadata.json         # you provide this
raw_videos/                  # downloaded .mp4s (gitignored)
candidates/                  # output (gitignored)
  continental/ eastern/ semi_western/ western/
  conflicted/
  unclear/
  manifest.csv
  processed_videos.json      # idempotency tracking, not for human review
pipeline/                    # pipeline modules
main.py                      # entry point
```

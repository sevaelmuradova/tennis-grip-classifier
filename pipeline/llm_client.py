"""Gemini API verification/labeling of hand-filtered frames.

Rate-limited to GEMINI_MAX_REQUESTS_PER_MINUTE (conservative default for the
free tier), retries once on any failure (network, rate limit, malformed
JSON) and then reports as "unclear"."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from pipeline.config import GEMINI_MAX_REQUESTS_PER_MINUTE, GEMINI_MODEL, LLM_MAX_RETRIES, LLM_MAX_TOKENS

logger = logging.getLogger("pipeline.llm_client")

PROVISIONAL_PROMPT_TEMPLATE = (
    "This image shows a hand gripping a tennis racket. Based on the source video, "
    "this frame is expected to show a {grip} grip. Does the visible grip match? "
    'Reply in JSON: {{"matches_provisional": true|false, '
    '"actual_grip_if_different": "continental|eastern|semi_western|western|unclear", '
    '"confidence": "high|medium|low", '
    '"reasoning": "one sentence about the base knuckle position on the racket handle"}}.'
)

NO_PROVISIONAL_PROMPT = (
    "This image shows a hand gripping a tennis racket. Identify which grip type it "
    "shows: Continental, Eastern, Semi-Western, Western, or Unclear. Consider the "
    "position of the base knuckle of the index finger on the racket handle. "
    'Reply in JSON: {"grip": "...", "confidence": "high|medium|low", "reasoning": "one sentence"}.'
)


@dataclass
class LLMResult:
    """Normalized result regardless of which prompt variant was used.

    matches_provisional / actual_grip_if_different are only meaningful when a
    provisional label was supplied. `grip` is only meaningful when there was
    no provisional label. `failed` is True if both API attempts errored out.
    """

    raw: dict
    confidence: Optional[str]
    reasoning: str
    matches_provisional: Optional[bool] = None
    actual_grip_if_different: Optional[str] = None
    grip: Optional[str] = None
    failed: bool = False


class _RateLimiter:
    def __init__(self, max_per_minute: float):
        self._min_interval = 60.0 / max_per_minute
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))


class LLMClient:
    def __init__(self, model: str = GEMINI_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. Get a free key at "
                "https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`."
            )
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter(GEMINI_MAX_REQUESTS_PER_MINUTE)

    def _call(self, jpg_bytes: bytes, prompt: str) -> dict:
        self._limiter.wait()
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        text = response.text
        if not text:
            raise ValueError(f"empty response from Gemini (possible safety block): {response!r}")
        return _extract_json(text)

    def _call_with_retry(self, jpg_bytes: bytes, prompt: str) -> Optional[dict]:
        attempts = LLM_MAX_RETRIES + 1
        for attempt in range(attempts):
            try:
                return self._call(jpg_bytes, prompt)
            except Exception as e:  # noqa: BLE001 - SDK error hierarchy not pinned down; retry once regardless
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                backoff = 20.0 if is_rate_limit else 2.0
                logger.warning("    LLM call failed (attempt %d/%d): %s", attempt + 1, attempts, e)
                if attempt < attempts - 1:
                    time.sleep(backoff)
        return None

    def verify_with_provisional(self, jpg_bytes: bytes, provisional_grip: str) -> LLMResult:
        prompt = PROVISIONAL_PROMPT_TEMPLATE.format(grip=provisional_grip)
        parsed = self._call_with_retry(jpg_bytes, prompt)
        if parsed is None:
            return LLMResult(raw={}, confidence=None, reasoning="LLM call failed after retry", failed=True)
        return LLMResult(
            raw=parsed,
            confidence=parsed.get("confidence"),
            reasoning=parsed.get("reasoning", ""),
            matches_provisional=parsed.get("matches_provisional"),
            actual_grip_if_different=parsed.get("actual_grip_if_different"),
        )

    def label_without_provisional(self, jpg_bytes: bytes) -> LLMResult:
        parsed = self._call_with_retry(jpg_bytes, NO_PROVISIONAL_PROMPT)
        if parsed is None:
            return LLMResult(raw={}, confidence=None, reasoning="LLM call failed after retry", failed=True)
        grip = (parsed.get("grip") or "").strip().lower().replace("-", "_").replace(" ", "_")
        return LLMResult(
            raw=parsed,
            confidence=parsed.get("confidence"),
            reasoning=parsed.get("reasoning", ""),
            grip=grip,
        )

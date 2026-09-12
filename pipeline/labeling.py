"""Final label assignment: combines the provisional (metadata-derived) label
with the LLM's verification/labeling result."""

from __future__ import annotations

from typing import Optional

from pipeline.config import VALID_GRIPS
from pipeline.llm_client import LLMResult

UNCLEAR = "unclear"
CONFLICTED = "conflicted"


def assign_final_label(
    provisional_grip: Optional[str],
    llm_result: LLMResult,
) -> tuple[str, bool, str]:
    """Return (final_label, needs_review, llm_label_for_manifest)."""

    if llm_result.failed:
        return UNCLEAR, True, UNCLEAR

    if provisional_grip is not None:
        matches = llm_result.matches_provisional
        confidence = llm_result.confidence

        if matches is True:
            llm_label = provisional_grip
            if confidence == "high":
                return provisional_grip, False, llm_label
            if confidence in ("medium", "low"):
                return provisional_grip, True, llm_label
            # malformed/missing confidence: be conservative and flag for review
            return provisional_grip, True, llm_label

        if matches is False:
            llm_label = llm_result.actual_grip_if_different or UNCLEAR
            return CONFLICTED, False, llm_label

        # matches_provisional missing/malformed
        return UNCLEAR, True, UNCLEAR

    # no provisional label (unlabeled-type video)
    grip = llm_result.grip
    confidence = llm_result.confidence
    llm_label = grip if grip in VALID_GRIPS else UNCLEAR

    if confidence == "high" and grip in VALID_GRIPS:
        return grip, False, llm_label
    return UNCLEAR, False, llm_label

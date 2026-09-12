"""MediaPipe Hands based frame filtering: keep frames with exactly one
confidently-detected hand whose bounding box is large enough in frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp

from pipeline.config import HAND_MIN_BBOX_AREA_RATIO, HAND_MIN_DETECTION_CONFIDENCE


@dataclass
class HandDetection:
    confidence: float
    bbox_area_ratio: float


class HandFilter:
    def __init__(
        self,
        min_detection_confidence: float = HAND_MIN_DETECTION_CONFIDENCE,
        min_bbox_area_ratio: float = HAND_MIN_BBOX_AREA_RATIO,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        )

    def check(self, frame_bgr) -> Optional[HandDetection]:
        """Return a HandDetection if `frame_bgr` passes the filter, else None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hand_landmarks = results.multi_hand_landmarks
        handedness = results.multi_handedness
        if not hand_landmarks or len(hand_landmarks) != 1:
            return None

        confidence = handedness[0].classification[0].score if handedness else 0.0
        if confidence <= self.min_detection_confidence:
            return None

        xs = [lm.x for lm in hand_landmarks[0].landmark]
        ys = [lm.y for lm in hand_landmarks[0].landmark]
        bbox_area_ratio = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if bbox_area_ratio < self.min_bbox_area_ratio:
            return None

        return HandDetection(confidence=confidence, bbox_area_ratio=bbox_area_ratio)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandFilter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

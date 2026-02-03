"""Person/face detector using MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float


class FaceDetector:
    """Detects faces in an image using MediaPipe."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """Return detections for a BGR image."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(image_rgb)
        if not results.detections:
            return []

        height, width = image_bgr.shape[:2]
        detections: List[Detection] = []
        for detection in results.detections:
            score = float(detection.score[0])
            if score < self.min_confidence:
                continue
            box = detection.location_data.relative_bounding_box
            x_min = int(max(box.xmin * width, 0))
            y_min = int(max(box.ymin * height, 0))
            box_width = int(min(box.width * width, width - x_min))
            box_height = int(min(box.height * height, height - y_min))
            detections.append(Detection(bbox=(x_min, y_min, box_width, box_height), confidence=score))
        return detections

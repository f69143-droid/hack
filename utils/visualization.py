"""Visualization helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from detection.person_detector import Detection
from model.gender_model import GenderPrediction


def draw_detections(
    image_bgr: np.ndarray,
    detections: Iterable[Detection],
    predictions: Iterable[GenderPrediction],
    show_boxes: bool = True,
) -> np.ndarray:
    output = image_bgr.copy()
    if not show_boxes:
        return output

    for detection, prediction in zip(detections, predictions):
        x, y, w, h = detection.bbox
        label = f"{prediction.label} ({prediction.confidence:.2f})"
        color = (255, 0, 0) if prediction.label == "Male" else (255, 105, 180)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            label,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def summarize_predictions(predictions: Iterable[GenderPrediction]) -> dict:
    summary = {"Male": 0, "Female": 0, "Unknown": 0}
    for prediction in predictions:
        summary[prediction.label] = summary.get(prediction.label, 0) + 1
    summary["Total"] = sum(summary.values())
    return summary

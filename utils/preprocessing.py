"""Image preprocessing utilities."""

from __future__ import annotations

import cv2
import numpy as np


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def crop_face(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    cropped = image_bgr[y : y + h, x : x + w]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def safe_resize(image_bgr: np.ndarray, max_width: int = 900) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr
    scale = max_width / width
    resized = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
    return resized

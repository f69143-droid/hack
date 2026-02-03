"""Gender classification model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


@dataclass
class GenderPrediction:
    label: str
    confidence: float


class GenderClassifier:
    """Wrapper for a PyTorch-based gender classifier.

    If weights are unavailable, a lightweight heuristic is used to keep the demo
    functional. Replace `weights_path` with a trained checkpoint for best results.
    """

    def __init__(self, weights_path: str | Path = "model/weights/gender_resnet18.pth") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.to(self.device)
        self.model.eval()

        self.weights_loaded = False
        weights_path = Path(weights_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.weights_loaded = True

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _heuristic_predict(self, image: np.ndarray) -> GenderPrediction:
        """Fallback heuristic when weights are not available."""

        hsv = Image.fromarray(image).convert("HSV")
        hsv_np = np.array(hsv)
        saturation = float(np.mean(hsv_np[:, :, 1]))
        label = "Female" if saturation > 90 else "Male"
        confidence = min(0.6 + (abs(saturation - 90) / 200), 0.75)
        return GenderPrediction(label=label, confidence=confidence)

    def predict(self, face_image: np.ndarray) -> GenderPrediction:
        """Predict gender from a face crop in RGB format."""

        if face_image.size == 0:
            return GenderPrediction(label="Unknown", confidence=0.0)

        if not self.weights_loaded:
            return self._heuristic_predict(face_image)

        pil_image = Image.fromarray(face_image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label_index = int(np.argmax(probs))
        label = "Male" if label_index == 0 else "Female"
        confidence = float(probs[label_index])
        return GenderPrediction(label=label, confidence=confidence)


def batch_predict(classifier: GenderClassifier, faces: list[np.ndarray]) -> list[GenderPrediction]:
    """Run prediction for a list of face crops."""

    return [classifier.predict(face) for face in faces]

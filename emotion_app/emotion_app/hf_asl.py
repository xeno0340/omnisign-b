from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class HFResult:
    label: str
    score: float


class HFASLClassifier:
    """
    Optional Hugging Face image classifier for ASL fingerspelling (static A–Z).

    - Runs locally once the model is downloaded (first run downloads).
    - Expects a cropped hand image (BGR numpy array).
    """

    def __init__(self, model_id: str = "aalof/clipvision-asl-fingerspelling") -> None:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Hugging Face mode requires extra packages.\n"
                "Install:\n"
                "  python -m pip install -U transformers torch torchvision pillow\n"
            ) from e

        # device: keep default auto; users can optimize later
        self._pipe = pipeline("image-classification", model=model_id)

    def predict(self, hand_bgr: np.ndarray) -> Optional[HFResult]:
        if hand_bgr is None or hand_bgr.size == 0:
            return None

        # Convert BGR -> RGB (pipeline accepts PIL or numpy RGB)
        rgb = hand_bgr[:, :, ::-1]
        try:
            out = self._pipe(rgb, top_k=1)
        except Exception:
            return None

        if not out:
            return None
        best = out[0]
        label = str(best.get("label", "")).strip()
        score = float(best.get("score", 0.0))

        # Normalize label: many models return "A".."Z" already
        if len(label) > 1:
            # Some pipelines return "LABEL_0"; we can't map reliably without config.
            # Fallback: keep raw label.
            return HFResult(label=label, score=score)
        return HFResult(label=label.upper(), score=score)


def crop_hand_from_landmarks(frame_bgr: np.ndarray, hand_landmarks, pad: int = 18) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    if not xs or not ys:
        return None
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame_bgr[y0:y1, x0:x1].copy()


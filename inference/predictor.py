"""High-level predictor used by API and CLI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inference.pix2pix_inference import Pix2PixInference


class ColorizationPredictor:
    def __init__(
        self,
        checkpoint: Path | str,
        device: str = "auto",
        image_size: tuple[int, int] | None = None,
    ):
        self.engine = Pix2PixInference(
            checkpoint,
            device=device,
            image_size=image_size,
        )

    def predict(self, image_path: Path | str) -> bytes:
        """Return colorized image as PNG bytes."""
        rgb = self.engine.colorize_file(image_path)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode colorized image.")
        return encoded.tobytes()

    def predict_array(self, thermal: np.ndarray) -> np.ndarray:
        return self.engine.colorize_array(thermal)

"""Thermal-to-RGB colorization using Pix2Pix with OpenCV fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from inference.pix2pix_inference import Pix2PixInference

logger = logging.getLogger(__name__)


class ColorizationService:
    def __init__(self, *, checkpoint: Path | None = None, device: str = "auto"):
        self.device = device
        self.engine: Pix2PixInference | None = None
        self.uses_neural = False

        if checkpoint and checkpoint.exists():
            self.engine = Pix2PixInference(checkpoint, device=device)
            self.uses_neural = True
            logger.info("Loaded Pix2Pix colorization checkpoint from %s", checkpoint)
        else:
            logger.info("No Pix2Pix checkpoint found; using OpenCV inferno colormap fallback")

    def colorize(self, thermal: np.ndarray) -> np.ndarray:
        if self.uses_neural and self.engine is not None:
            return self.engine.colorize_array(thermal)

        normalized = self._normalize(thermal)
        gray = (normalized * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)

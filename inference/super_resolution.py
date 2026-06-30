"""Super-resolution inference using PyTorch with OpenCV fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SuperResolutionNet(nn.Module):
    """Lightweight SR network for 2x upsampling."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.body = nn.Sequential(_ResidualBlock(64), _ResidualBlock(64), _ResidualBlock(64))
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64 * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.head(x))
        x = self.body(x)
        return self.tail(x)


class SuperResolutionService:
    def __init__(self, *, scale: int = 2, checkpoint: Path | None = None, device: str = "auto"):
        self.scale = scale
        self.device = self._resolve_device(device)
        self.model = SuperResolutionNet(scale=scale).to(self.device)
        self.uses_neural = False

        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self.uses_neural = True
            logger.info("Loaded super-resolution checkpoint from %s", checkpoint)
        else:
            self.model.eval()
            logger.info(
                "No SR checkpoint found; using OpenCV Lanczos upsampling (scale=%sx)",
                scale,
            )

    def upscale(self, thermal: np.ndarray) -> np.ndarray:
        normalized = self._normalize(thermal)

        if self.uses_neural:
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                output = self.model(tensor).squeeze().cpu().numpy()
            return self._rescale_to_original_range(output, thermal)

        height, width = normalized.shape
        upscaled = cv2.resize(
            normalized,
            (width * self.scale, height * self.scale),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return self._rescale_to_original_range(upscaled, thermal)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_val) / (max_val - min_val)

    @staticmethod
    def _rescale_to_original_range(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ref = reference.astype(np.float32)
        min_val = float(ref.min())
        max_val = float(ref.max())
        output = output.astype(np.float32)
        out_min = float(output.min())
        out_max = float(output.max())
        if out_max - out_min < 1e-6:
            return np.full_like(output, min_val, dtype=np.float32)
        scaled = (output - out_min) / (out_max - out_min)
        return scaled * (max_val - min_val) + min_val

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

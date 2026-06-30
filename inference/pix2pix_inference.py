"""Pix2Pix inference for thermal-to-RGB colorization."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.preprocessing import (
    denormalize_rgb_tensor,
    load_thermal_array,
    normalize_thermal,
    resize_pair,
    tensorize_thermal,
)
from models.architectures.pix2pix import Pix2PixConfig
from models.architectures.unet_generator import UNetGenerator

logger = logging.getLogger(__name__)


class Pix2PixInference:
    """Load a trained Pix2Pix generator and colorize 100m thermal imagery."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str = "auto",
        image_size: tuple[int, int] | None = None,
    ):
        self.checkpoint = Path(checkpoint)
        self.device = self._resolve_device(device)
        self.image_size = image_size
        self.generator = self._load_generator()

    def _load_generator(self) -> UNetGenerator:
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)

        if isinstance(payload, dict) and "generator" in payload:
            state_dict = payload["generator"]
            config = Pix2PixConfig(**payload.get("model_config", {}))
        elif isinstance(payload, dict):
            state_dict = payload
            config = Pix2PixConfig()
        else:
            state_dict = payload
            config = Pix2PixConfig()

        generator = UNetGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            ngf=config.ngf,
            use_dropout=False,
        ).to(self.device)
        generator.load_state_dict(state_dict)
        generator.eval()
        logger.info("Loaded Pix2Pix generator from %s", self.checkpoint)
        return generator

    def colorize_array(self, thermal: np.ndarray) -> np.ndarray:
        normalized = normalize_thermal(thermal.astype(np.float32))
        if self.image_size is not None:
            dummy_rgb = np.zeros((*self.image_size, 3), dtype=np.float32)
            normalized, _ = resize_pair(normalized, dummy_rgb, self.image_size)

        tensor = tensorize_thermal(normalized).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.generator(tensor).squeeze(0)
        return denormalize_rgb_tensor(output)

    def colorize_file(self, input_path: Path | str, output_path: Path | str | None = None) -> np.ndarray:
        input_path = Path(input_path)
        thermal = load_thermal_array(input_path)
        rgb = self.colorize_array(thermal)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            logger.info("Saved colorized image to %s", output_path)

        return rgb

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

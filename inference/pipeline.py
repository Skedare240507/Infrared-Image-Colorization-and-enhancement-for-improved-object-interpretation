"""End-to-end thermal processing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from api.services.storage_service import ARTIFACT_FILES, ArtifactName
from api.services.tiff_service import TiffService
from inference.colorization import ColorizationService
from inference.super_resolution import SuperResolutionService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    super_resolution_path: Path
    colorized_path: Path
    preview_path: Path
    metadata: dict


class ProcessingPipeline:
    def __init__(
        self,
        *,
        tiff_service: TiffService,
        sr_service: SuperResolutionService,
        colorization_service: ColorizationService,
    ):
        self.tiff_service = tiff_service
        self.sr_service = sr_service
        self.colorization_service = colorization_service

    def run(self, input_path: Path, job_dir: Path) -> PipelineResult:
        logger.info("Starting pipeline for %s", input_path)
        thermal_image = self.tiff_service.read_thermal(input_path)

        upscaled = self.sr_service.upscale(thermal_image.array)
        sr_path = job_dir / ARTIFACT_FILES[ArtifactName.SUPER_RESOLUTION]
        updated_profile = thermal_image.profile.copy()
        updated_profile.update(
            {
                "height": upscaled.shape[0],
                "width": upscaled.shape[1],
                "transform": thermal_image.profile.get("transform"),
            }
        )
        self.tiff_service.write_geotiff(upscaled, updated_profile, sr_path)

        colorized = self.colorization_service.colorize(upscaled)
        colorized_path = job_dir / ARTIFACT_FILES[ArtifactName.COLORIZED]
        cv2.imwrite(str(colorized_path), cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))

        preview_path = job_dir / ARTIFACT_FILES[ArtifactName.PREVIEW]
        self._save_preview(upscaled, colorized, preview_path)

        metadata = {
            "input_shape": [thermal_image.height, thermal_image.width],
            "output_shape": [upscaled.shape[0], upscaled.shape[1]],
            "sr_scale": self.sr_service.scale,
            "sr_backend": "pytorch" if self.sr_service.uses_neural else "opencv",
            "colorization_backend": (
                "pytorch" if self.colorization_service.uses_neural else "opencv_colormap"
            ),
        }
        logger.info("Pipeline completed for %s", input_path)
        return PipelineResult(
            super_resolution_path=sr_path,
            colorized_path=colorized_path,
            preview_path=preview_path,
            metadata=metadata,
        )

    @staticmethod
    def _save_preview(thermal: np.ndarray, colorized: np.ndarray, destination: Path) -> None:
        normalized = thermal.astype(np.float32)
        min_val = float(normalized.min())
        max_val = float(normalized.max())
        if max_val - min_val > 1e-6:
            normalized = (normalized - min_val) / (max_val - min_val)
        gray = (normalized * 255.0).astype(np.uint8)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if gray_rgb.shape[:2] != colorized.shape[:2]:
            gray_rgb = cv2.resize(
                gray_rgb,
                (colorized.shape[1], colorized.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        preview = np.hstack([gray_rgb, colorized])
        cv2.imwrite(str(destination), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

"""Thermal TIFF validation and I/O via Rasterio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2  # BUG-FIX: move to module level - avoids repeated import overhead & ImportError late-surprise
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

from api.errors.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermalImage:
    array: np.ndarray
    profile: dict
    width: int
    height: int
    band_count: int


class TiffService:
    def validate_upload(self, path: Path, allowed_extensions: frozenset[str]) -> None:
        suffix = path.suffix.lower()
        if suffix not in allowed_extensions:
            raise ValidationError(
                "Invalid file type. Only thermal TIFF and PNG uploads are supported.",
                details={"allowed_extensions": sorted(allowed_extensions)},
            )

        if not path.exists() or path.stat().st_size == 0:
            raise ValidationError("Uploaded file is missing or empty.")

        if suffix in {".tif", ".tiff"}:
            try:
                with rasterio.open(path) as src:
                    if src.count < 1:
                        # BUG-FIX: was not raising inside try block correctly before refactor
                        raise ValidationError("TIFF contains no image bands.")
                    if src.width < 16 or src.height < 16:
                        raise ValidationError(
                            "TIFF dimensions are too small for processing.",
                            details={"width": src.width, "height": src.height},
                        )
            except ValidationError:
                raise  # BUG-FIX: re-raise ValidationError explicitly so it is NOT swallowed by RasterioIOError handler
            except RasterioIOError as exc:
                logger.exception("Rasterio failed to open TIFF: %s", path)
                raise ValidationError(
                    "Unable to read thermal TIFF.",
                    details={"reason": str(exc)},
                ) from exc
        elif suffix == ".png":
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValidationError("Unable to read PNG image. File may be corrupt or unsupported colour depth.")
            if img.shape[1] < 16 or img.shape[0] < 16:
                raise ValidationError(
                    "PNG dimensions are too small for processing.",
                    details={"width": img.shape[1], "height": img.shape[0]},
                )

    def read_thermal(self, path: Path) -> ThermalImage:
        self.validate_upload(path, frozenset({".tif", ".tiff", ".png"}))

        suffix = path.suffix.lower()
        if suffix == ".png":
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # BUG-FIX: validate_upload already checked for None, but guard again since
            # files can be deleted between validation and read on concurrent requests
            if raw is None:
                raise ValidationError("PNG file could not be read from disk.")
            thermal = raw.astype(np.float32)
            profile = {
                "driver": "GTiff",
                "height": thermal.shape[0],
                "width": thermal.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": from_origin(77.58, 12.97, 0.0001, 0.0001),
            }
            logger.info("Loaded thermal PNG image %s (%dx%d)", path.name, thermal.shape[1], thermal.shape[0])
            return ThermalImage(
                array=thermal,
                profile=profile,
                width=thermal.shape[1],
                height=thermal.shape[0],
                band_count=1,
            )

        with rasterio.open(path) as src:
            thermal = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            logger.info(
                "Loaded thermal TIFF %s (%dx%d, %d band(s))",
                path.name,
                src.width,
                src.height,
                src.count,
            )
            return ThermalImage(
                array=thermal,
                profile=profile,
                width=src.width,
                height=src.height,
                band_count=src.count,
            )

    def write_geotiff(self, array: np.ndarray, profile: dict, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_profile = profile.copy()

        # BUG-FIX: PNG-sourced profiles use string CRS ("EPSG:4326") which rasterio
        # accepts fine, but we must ensure the driver is always 'GTiff' so writing
        # doesn't fail with 'Cannot create driver PNG as GTiff'.
        output_profile["driver"] = "GTiff"

        output_profile.update(
            {
                "dtype": rasterio.float32 if array.dtype != np.uint8 else rasterio.uint8,
                "count": 1 if array.ndim == 2 else array.shape[0],
                "height": array.shape[-2],
                "width": array.shape[-1],
            }
        )

        data = array[np.newaxis, ...] if array.ndim == 2 else array
        with rasterio.open(destination, "w", **output_profile) as dst:
            dst.write(data.astype(output_profile["dtype"]))
        return destination

"""Image loading and normalization for 100m thermal and RGB pairs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_thermal_array(path: Path) -> np.ndarray:
    """Load the first band from a 100m thermal GeoTIFF."""
    try:
        with rasterio.open(path) as src:
            thermal = src.read(1).astype(np.float32)
    except RasterioIOError:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read thermal image: {path}")
        thermal = image.astype(np.float32)
        if thermal.ndim == 3:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)

    return normalize_thermal(thermal)


def load_rgb_array(path: Path) -> np.ndarray:
    """Load an RGB reference image."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            if src.count >= 3:
                rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
            else:
                band = src.read(1).astype(np.float32)
                rgb = np.stack([band, band, band], axis=-1)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read RGB image: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    return normalize_rgb(rgb)


def normalize_thermal(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized * 2.0 - 1.0


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    return array * 2.0 - 1.0


def resize_pair(
    thermal: np.ndarray,
    rgb: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    thermal_resized = cv2.resize(thermal, (width, height), interpolation=cv2.INTER_AREA)
    rgb_resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return thermal_resized, rgb_resized


def tensorize_thermal(array: np.ndarray):
    import torch

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    return torch.from_numpy(array.astype(np.float32))


def tensorize_rgb(array: np.ndarray):
    import torch

    return torch.from_numpy(np.transpose(array, (2, 0, 1)).astype(np.float32))


def denormalize_rgb_tensor(tensor) -> np.ndarray:
    import torch

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    rgb = ((array + 1.0) * 0.5).clip(0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)

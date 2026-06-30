"""Paired augmentations for thermal and RGB images."""

from __future__ import annotations

import random

import cv2
import numpy as np


def random_flip(thermal: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        thermal = np.fliplr(thermal)
        rgb = np.fliplr(rgb)
    if random.random() < 0.5:
        thermal = np.flipud(thermal)
        rgb = np.flipud(rgb)
    return thermal.copy(), rgb.copy()


def random_crop(
    thermal: np.ndarray,
    rgb: np.ndarray,
    crop_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = thermal.shape[:2]
    crop_h, crop_w = crop_size
    if height < crop_h or width < crop_w:
        return thermal, rgb

    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
    thermal_crop = thermal[top : top + crop_h, left : left + crop_w]
    rgb_crop = rgb[top : top + crop_h, left : left + crop_w]
    return thermal_crop, rgb_crop


def augment_pair(
    thermal: np.ndarray,
    rgb: np.ndarray,
    *,
    crop_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    thermal, rgb = random_flip(thermal, rgb)
    if crop_size is not None:
        thermal, rgb = random_crop(thermal, rgb, crop_size)
    return thermal, rgb

"""Paired 100m thermal and RGB dataset for Pix2Pix training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataset.augmentation import augment_pair
from dataset.preprocessing import (
    RGB_EXTENSIONS,
    load_rgb_array,
    load_thermal_array,
    resize_pair,
    tensorize_rgb,
    tensorize_thermal,
)


class ThermalRGBDataset(Dataset):
    """
    Paired thermal/RGB dataset.

    Expected layout:
        root/
          thermal/
            scene_001.tif
          rgb/
            scene_001.png
    """

    THERMAL_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: Path | str,
        *,
        split: str | None = None,
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        self.root = Path(root)
        if split:
            self.root = self.root / split

        self.thermal_dir = self.root / "thermal"
        self.rgb_dir = self.root / "rgb"
        self.image_size = image_size
        self.augment = augment

        if not self.thermal_dir.exists() or not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"Dataset folders not found under {self.root}. "
                "Expected 'thermal/' and 'rgb/' subdirectories."
            )

        self.pairs = self._discover_pairs()
        if not self.pairs:
            raise ValueError(f"No paired samples found in {self.root}")

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for thermal_path in sorted(self.thermal_dir.iterdir()):
            if thermal_path.suffix.lower() not in self.THERMAL_EXTENSIONS:
                continue
            rgb_path = self._find_rgb_pair(thermal_path.stem)
            if rgb_path is not None:
                pairs.append((thermal_path, rgb_path))
        return pairs

    def _find_rgb_pair(self, stem: str) -> Path | None:
        for ext in RGB_EXTENSIONS:
            candidate = self.rgb_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        thermal_path, rgb_path = self.pairs[index]
        thermal = load_thermal_array(thermal_path)
        rgb = load_rgb_array(rgb_path)

        thermal, rgb = resize_pair(thermal, rgb, self.image_size)

        if self.augment:
            thermal, rgb = augment_pair(thermal, rgb, crop_size=self.image_size)

        return {
            "thermal": tensorize_thermal(thermal),
            "rgb": tensorize_rgb(rgb),
            "stem": thermal_path.stem,
        }


def create_dataloader(
    root: Path | str,
    *,
    split: str = "train",
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool | None = None,
) -> DataLoader:
    if augment is None:
        augment = split == "train"

    dataset = ThermalRGBDataset(
        root,
        split=split,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )

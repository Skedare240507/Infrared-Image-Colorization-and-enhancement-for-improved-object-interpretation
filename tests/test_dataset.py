from pathlib import Path

import pytest

from dataset.loaders import ThermalRGBDataset


def test_dataset_length(sample_pair_dirs: Path):
    dataset = ThermalRGBDataset(sample_pair_dirs, split="train", image_size=(256, 256))
    assert len(dataset) == 1

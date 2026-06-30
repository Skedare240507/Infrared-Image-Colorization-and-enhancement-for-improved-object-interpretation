"""Pytest fixtures for API integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from api.app import create_app
from api.config import AppConfig


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        secret_key="test-secret",
        debug=True,
        max_upload_mb=10,
        allowed_extensions=frozenset({".tif", ".tiff", ".png"}),
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        log_level="WARNING",
        sr_scale=2,
        sr_checkpoint=None,
        colorization_checkpoint=None,
        device="cpu",
    )


@pytest.fixture
def app(app_config: AppConfig):
    flask_app = create_app(app_config)
    flask_app.config.update({"TESTING": True})
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "thermal.tif"
    data = (np.linspace(0, 255, 64 * 64, dtype=np.float32).reshape(64, 64))
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 64, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def sample_pair_dirs(tmp_path: Path):
    split_root = tmp_path / "train"
    thermal_dir = split_root / "thermal"
    rgb_dir = split_root / "rgb"
    thermal_dir.mkdir(parents=True)
    rgb_dir.mkdir(parents=True)

    size = 256
    thermal = np.linspace(0, 300, size * size, dtype=np.float32).reshape(size, size)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, size, 1, 1),
    }
    with rasterio.open(thermal_dir / "scene_001.tif", "w", **profile) as dst:
        dst.write(thermal, 1)

    rgb = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    import cv2

    cv2.imwrite(str(rgb_dir / "scene_001.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return tmp_path

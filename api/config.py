"""Flask application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from utils.config import load_config


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    secret_key: str
    debug: bool
    max_upload_mb: int
    allowed_extensions: frozenset[str]
    upload_dir: Path
    output_dir: Path
    log_dir: Path
    log_level: str
    sr_scale: int
    sr_checkpoint: Path | None
    colorization_checkpoint: Path | None
    device: str

    @property
    def max_content_length(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _resolve_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = (ROOT_DIR / path).resolve()
    return resolved if resolved.exists() else None


def load_app_config() -> AppConfig:
    yaml_config = load_config(ROOT_DIR / "configs" / "default.yaml")
    api_cfg = yaml_config.get("api", {})
    inference_cfg = yaml_config.get("inference", {})

    upload_dir = ROOT_DIR / api_cfg.get("upload_dir", "uploads")
    output_dir = ROOT_DIR / api_cfg.get("output_dir", "outputs")
    log_dir = ROOT_DIR / api_cfg.get("log_dir", "logs")

    return AppConfig(
        secret_key=os.getenv("FLASK_SECRET_KEY", "change-me-in-production"),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", api_cfg.get("max_upload_mb", 50))),
        allowed_extensions=frozenset(
            ext.lower()
            for ext in api_cfg.get("allowed_extensions", [".tif", ".tiff"])
        ),
        upload_dir=upload_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        log_level=os.getenv("LOG_LEVEL", api_cfg.get("log_level", "INFO")),
        sr_scale=int(os.getenv("SR_SCALE", api_cfg.get("sr_scale", 2))),
        sr_checkpoint=_resolve_checkpoint(
            os.getenv("SR_CHECKPOINT", inference_cfg.get("sr_checkpoint"))
        ),
        colorization_checkpoint=_resolve_checkpoint(
            os.getenv(
                "COLORIZATION_CHECKPOINT",
                inference_cfg.get("checkpoint", "models/weights/best.pt"),
            )
        ),
        device=os.getenv("INFERENCE_DEVICE", inference_cfg.get("device", "auto")),
    )

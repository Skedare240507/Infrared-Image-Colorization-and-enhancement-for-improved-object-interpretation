"""CLI for Pix2Pix thermal colorization inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inference.pix2pix_inference import Pix2PixInference
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pix2Pix thermal colorization inference")
    parser.add_argument("--input", type=Path, required=True, help="Thermal TIFF/image path")
    parser.add_argument("--output", type=Path, required=True, help="Output RGB PNG path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/weights/colorization.pt"),
        help="Generator or full Pix2Pix checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir=Path("logs"), level="INFO")

    engine = Pix2PixInference(
        args.checkpoint,
        device=args.device,
        image_size=tuple(args.image_size) if args.image_size else None,
    )
    engine.colorize_file(args.input, args.output)
    logger.info("Inference complete: %s", args.output)


if __name__ == "__main__":
    main()

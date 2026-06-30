"""Evaluate Pix2Pix generator on a validation split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.loaders import create_dataloader
from dataset.preprocessing import denormalize_rgb_tensor
from models.architectures.pix2pix import Pix2PixConfig, Pix2PixModel
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    data_root: Path,
    *,
    split: str = "val",
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 4,
    device: str = "auto",
    output_dir: Path | None = None,
) -> dict[str, float]:
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model_config = Pix2PixConfig(**payload.get("model_config", {}))
    model = Pix2PixModel(model_config).to(torch_device)
    model.generator.load_state_dict(payload["generator"])
    model.eval()

    loader = create_dataloader(
        data_root,
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    l1 = torch.nn.L1Loss()
    total_l1 = 0.0
    batches = 0

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        thermal = batch["thermal"].to(torch_device)
        target_rgb = batch["rgb"].to(torch_device)
        fake_rgb = model.generate(thermal)
        total_l1 += float(l1(fake_rgb, target_rgb).item())
        batches += 1

        if output_dir:
            import cv2

            for idx in range(fake_rgb.shape[0]):
                image = denormalize_rgb_tensor(fake_rgb[idx])
                stem = batch["stem"][idx]
                cv2.imwrite(
                    str(output_dir / f"{stem}_pred.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )

    metrics = {"l1": total_l1 / max(batches, 1), "samples": len(loader.dataset)}
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    setup_logging(log_dir=Path("logs"), level="INFO")
    evaluate(
        args.checkpoint,
        args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

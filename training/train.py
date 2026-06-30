"""Pix2Pix training entrypoint."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from dataset.loaders import create_dataloader
from models.architectures.pix2pix import Pix2PixConfig, Pix2PixModel
from training.pix2pix_trainer import Pix2PixTrainer
from utils.config import load_config
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pix2Pix thermal colorization model")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})

    setup_logging(log_dir=Path("logs"), level="INFO")
    set_seed(project_cfg.get("seed", 42))

    image_size = tuple(args.image_size or data_cfg.get("image_size", [256, 256]))
    batch_size = args.batch_size or data_cfg.get("batch_size", 4)
    epochs = args.epochs or train_cfg.get("epochs", 200)
    checkpoint_dir = args.checkpoint_dir or Path(train_cfg.get("checkpoint_dir", "models/weights"))
    device = resolve_device(args.device)

    pix2pix_config = Pix2PixConfig(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 3),
        ngf=model_cfg.get("ngf", 64),
        ndf=model_cfg.get("ndf", 64),
        n_layers=model_cfg.get("n_layers", 3),
        use_dropout=model_cfg.get("use_dropout", True),
    )
    model = Pix2PixModel(pix2pix_config)

    train_loader = create_dataloader(
        args.data_root,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=True,
        augment=True,
    )

    try:
        val_loader = create_dataloader(
            args.data_root,
            split="val",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=False,
            augment=False,
        )
    except (FileNotFoundError, ValueError):
        logger.warning("Validation split unavailable; training without validation.")
        val_loader = None

    trainer = Pix2PixTrainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        lr=float(train_cfg.get("lr", model_cfg.get("lr", 2e-4))),
        beta1=float(train_cfg.get("beta1", model_cfg.get("beta1", 0.5))),
        lambda_l1=float(train_cfg.get("lambda_l1", model_cfg.get("lambda_l1", 100.0))),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info(
        "Starting Pix2Pix training on %s with %s train samples",
        device,
        len(train_loader.dataset),
    )
    trainer.fit(epochs=epochs, save_every=int(train_cfg.get("save_every", 10)))


if __name__ == "__main__":
    main()

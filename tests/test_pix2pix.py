"""Pix2Pix model and pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dataset.loaders import ThermalRGBDataset, create_dataloader
from inference.pix2pix_inference import Pix2PixInference
from models.architectures.pix2pix import Pix2PixConfig, Pix2PixModel
from training.losses.gan_loss import Pix2PixLoss
from training.pix2pix_trainer import Pix2PixTrainer


def test_pix2pix_forward_pass():
    model = Pix2PixModel(Pix2PixConfig())
    thermal = torch.randn(2, 1, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)

    fake_rgb = model.generate(thermal)
    assert fake_rgb.shape == (2, 3, 256, 256)
    assert fake_rgb.min() >= -1.0
    assert fake_rgb.max() <= 1.0

    pred_real = model.discriminator(thermal, rgb)
    pred_fake = model.discriminator(thermal, fake_rgb.detach())
    assert pred_real.shape[0] == 2
    assert pred_fake.shape[0] == 2


def test_gan_losses():
    model = Pix2PixModel(Pix2PixConfig())
    loss_fn = Pix2PixLoss(lambda_l1=100.0)
    thermal = torch.randn(1, 1, 256, 256)
    target_rgb = torch.randn(1, 3, 256, 256)
    fake_rgb = model.generate(thermal)

    pred_real = model.discriminator(thermal, target_rgb)
    pred_fake = model.discriminator(thermal, fake_rgb.detach())
    loss_d = loss_fn.discriminator_loss(pred_real, pred_fake)

    pred_fake_for_g = model.discriminator(thermal, fake_rgb)
    loss_g, adv, l1 = loss_fn.generator_loss(pred_fake_for_g, fake_rgb, target_rgb)

    assert loss_d.item() >= 0
    assert loss_g.item() >= 0
    assert adv.item() >= 0
    assert l1.item() >= 0


def test_dataset_loader(sample_pair_dirs: Path):
    dataset = ThermalRGBDataset(sample_pair_dirs, split="train", image_size=(256, 256))
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["thermal"].shape == (1, 256, 256)
    assert sample["rgb"].shape == (3, 256, 256)
    assert sample["thermal"].min() >= -1.0
    assert sample["thermal"].max() <= 1.0


def test_trainer_single_step(sample_pair_dirs: Path):
    loader = create_dataloader(
        sample_pair_dirs,
        split="train",
        image_size=(256, 256),
        batch_size=1,
        num_workers=0,
        shuffle=False,
        augment=False,
    )
    model = Pix2PixModel(Pix2PixConfig(ngf=32, ndf=32, use_dropout=False))
    trainer = Pix2PixTrainer(
        model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        checkpoint_dir=sample_pair_dirs / "checkpoints",
        lambda_l1=10.0,
    )
    metrics = trainer.train_epoch()
    assert "loss_g" in metrics
    assert metrics["loss_g"] >= 0


def test_inference_from_generator_checkpoint(tmp_path: Path):
    config = Pix2PixConfig(ngf=32, use_dropout=False)
    model = Pix2PixModel(config)
    checkpoint = tmp_path / "colorization.pt"
    torch.save(
        {"generator": model.generator.state_dict(), "model_config": config.__dict__},
        checkpoint,
    )

    thermal = np.linspace(0, 255, 256 * 256, dtype=np.float32).reshape(256, 256)
    engine = Pix2PixInference(checkpoint, device="cpu", image_size=(256, 256))
    rgb = engine.colorize_array(thermal)

    assert rgb.shape == (256, 256, 3)
    assert rgb.dtype == np.uint8

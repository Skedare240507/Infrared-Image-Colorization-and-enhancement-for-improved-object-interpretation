"""Pix2Pix training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.architectures.pix2pix import Pix2PixConfig, Pix2PixModel
from training.losses.gan_loss import Pix2PixLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_val_l1: float = float("inf")


class Pix2PixTrainer:
    def __init__(
        self,
        model: Pix2PixModel,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        checkpoint_dir: Path,
        lr: float = 2e-4,
        beta1: float = 0.5,
        lambda_l1: float = 100.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = Pix2PixLoss(lambda_l1=lambda_l1)
        self.state = TrainState()

        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
        )
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals = {"loss_g": 0.0, "loss_d": 0.0, "loss_l1": 0.0, "batches": 0}

        for batch in self.train_loader:
            thermal = batch["thermal"].to(self.device)
            target_rgb = batch["rgb"].to(self.device)

            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_rgb = self.model.generate(thermal)

            pred_real = self.model.discriminator(thermal, target_rgb)
            pred_fake = self.model.discriminator(thermal, fake_rgb.detach())
            loss_d = self.loss_fn.discriminator_loss(pred_real, pred_fake)
            loss_d.backward()
            self.optimizer_d.step()

            self.optimizer_g.zero_grad(set_to_none=True)
            fake_rgb = self.model.generate(thermal)
            pred_fake = self.model.discriminator(thermal, fake_rgb)
            loss_g, _, loss_l1 = self.loss_fn.generator_loss(pred_fake, fake_rgb, target_rgb)
            loss_g.backward()
            self.optimizer_g.step()

            totals["loss_g"] += float(loss_g.item())
            totals["loss_d"] += float(loss_d.item())
            totals["loss_l1"] += float(loss_l1.item())
            totals["batches"] += 1
            self.state.step += 1

        count = max(totals["batches"], 1)
        return {
            "loss_g": totals["loss_g"] / count,
            "loss_d": totals["loss_d"] / count,
            "loss_l1": totals["loss_l1"] / count,
        }

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {"val_l1": 0.0}

        self.model.eval()
        total_l1 = 0.0
        batches = 0
        l1 = torch.nn.L1Loss()

        for batch in self.val_loader:
            thermal = batch["thermal"].to(self.device)
            target_rgb = batch["rgb"].to(self.device)
            fake_rgb = self.model.generate(thermal)
            total_l1 += float(l1(fake_rgb, target_rgb).item())
            batches += 1

        return {"val_l1": total_l1 / max(batches, 1)}

    def save_checkpoint(self, name: str, metrics: dict[str, float] | None = None) -> Path:
        payload = {
            "epoch": self.state.epoch,
            "step": self.state.step,
            "model_config": self.model.config.__dict__,
            "generator": self.model.generator.state_dict(),
            "discriminator": self.model.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "metrics": metrics or {},
            "best_val_l1": self.state.best_val_l1,
        }
        path = self.checkpoint_dir / name
        torch.save(payload, path)
        logger.info("Saved checkpoint to %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.generator.load_state_dict(checkpoint["generator"])
        self.model.discriminator.load_state_dict(checkpoint["discriminator"])
        if "optimizer_g" in checkpoint:
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if "optimizer_d" in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.step = checkpoint.get("step", 0)
        self.state.best_val_l1 = checkpoint.get("best_val_l1", float("inf"))
        logger.info("Loaded checkpoint from %s (epoch %s)", path, self.state.epoch)

    def fit(self, epochs: int, save_every: int = 10) -> None:
        for epoch in range(1, epochs + 1):
            self.state.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            logger.info(
                "Epoch %s/%s | G: %.4f | D: %.4f | L1: %.4f | val_L1: %.4f",
                epoch,
                epochs,
                train_metrics["loss_g"],
                train_metrics["loss_d"],
                train_metrics["loss_l1"],
                val_metrics["val_l1"],
            )

            metrics = {**train_metrics, **val_metrics}
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt", metrics)

            if val_metrics["val_l1"] < self.state.best_val_l1:
                self.state.best_val_l1 = val_metrics["val_l1"]
                self.save_checkpoint("best.pt", metrics)
                generator_path = self.checkpoint_dir / "colorization.pt"
                torch.save(
                    {
                        "generator": self.model.generator.state_dict(),
                        "model_config": self.model.config.__dict__,
                    },
                    generator_path,
                )
                logger.info("Updated best generator weights at %s", generator_path)

        self.save_checkpoint("last.pt", metrics)

"""Pix2Pix GAN and L1 losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Least-squares GAN loss used in the original Pix2Pix paper."""

    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real))
        self.register_buffer("fake_label", torch.tensor(target_fake))
        self.loss = nn.MSELoss()

    def _expand(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self.real_label if is_real else self.fake_label
        return self.loss(prediction, self._expand(prediction, target))


class Pix2PixLoss:
    """Combined adversarial and L1 reconstruction loss."""

    def __init__(self, lambda_l1: float = 100.0):
        self.lambda_l1 = lambda_l1
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        loss_real = self.gan_loss(real_pred, is_real=True)
        loss_fake = self.gan_loss(fake_pred, is_real=False)
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(
        self,
        fake_pred: torch.Tensor,
        fake_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adv_loss = self.gan_loss(fake_pred, is_real=True)
        l1_loss = self.l1_loss(fake_rgb, target_rgb) * self.lambda_l1
        total = adv_loss + l1_loss
        return total, adv_loss, l1_loss

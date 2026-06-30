"""Pix2Pix model wrapper combining generator and discriminator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.architectures.patchgan_discriminator import PatchGANDiscriminator
from models.architectures.unet_generator import UNetGenerator


@dataclass(frozen=True)
class Pix2PixConfig:
    in_channels: int = 1
    out_channels: int = 3
    ngf: int = 64
    ndf: int = 64
    n_layers: int = 3
    use_dropout: bool = True


class Pix2PixModel(nn.Module):
    """Pix2Pix conditional GAN for thermal-to-RGB translation."""

    def __init__(self, config: Pix2PixConfig | None = None):
        super().__init__()
        self.config = config or Pix2PixConfig()
        self.generator = UNetGenerator(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            ngf=self.config.ngf,
            use_dropout=self.config.use_dropout,
        )
        self.discriminator = PatchGANDiscriminator(
            in_channels=self.config.in_channels + self.config.out_channels,
            ndf=self.config.ndf,
            n_layers=self.config.n_layers,
        )

    def generate(self, thermal: torch.Tensor) -> torch.Tensor:
        return self.generator(thermal)

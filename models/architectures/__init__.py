"""Neural network architecture registry."""

from __future__ import annotations

import torch.nn as nn

from models.architectures.pix2pix import Pix2PixConfig, Pix2PixModel
from models.architectures.patchgan_discriminator import PatchGANDiscriminator
from models.architectures.unet_generator import UNetGenerator


def build_pix2pix(**kwargs) -> Pix2PixModel:
    config = Pix2PixConfig(
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 3),
        ngf=kwargs.get("ngf", 64),
        ndf=kwargs.get("ndf", 64),
        n_layers=kwargs.get("n_layers", 3),
        use_dropout=kwargs.get("use_dropout", True),
    )
    return Pix2PixModel(config)


def build_generator(**kwargs) -> UNetGenerator:
    return UNetGenerator(
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 3),
        ngf=kwargs.get("ngf", 64),
        use_dropout=kwargs.get("use_dropout", True),
    )


def build_discriminator(**kwargs) -> PatchGANDiscriminator:
    in_channels = kwargs.get("in_channels", 1) + kwargs.get("out_channels", 3)
    return PatchGANDiscriminator(
        in_channels=in_channels,
        ndf=kwargs.get("ndf", 64),
        n_layers=kwargs.get("n_layers", 3),
    )


def build_model(architecture: str, **kwargs) -> nn.Module:
    registry = {
        "pix2pix": build_pix2pix,
        "unet_generator": build_generator,
        "patchgan_discriminator": build_discriminator,
    }
    if architecture not in registry:
        raise NotImplementedError(f"Unknown architecture: {architecture}")
    return registry[architecture](**kwargs)

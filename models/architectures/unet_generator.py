"""Pix2Pix U-Net generator with skip connections."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """U-Net generator: thermal (1ch) -> RGB (3ch) with tanh output."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        ngf: int = 64,
        use_dropout: bool = False,
    ):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        num_downs = 8

        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            out_channels,
            ngf,
            input_nc=in_channels,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )
        self.model = unet_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        *,
        input_nc: int | None = None,
        submodule: UnetSkipConnectionBlock | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down_layers: list[nn.Module] = []
        up_layers: list[nn.Module] = []
        if outermost:
            down_layers.append(nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1))
            up_layers.extend(
                [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),
                ]
            )
        elif innermost:
            down_layers.extend([nn.LeakyReLU(0.2, True), nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)])
            up_layers.extend([nn.ReLU(True), nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)])
        else:
            down_layers.extend(
                [
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(inner_nc),
                ]
            )
            up_layers.extend(
                [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                ]
            )
            if use_dropout:
                up_layers.append(nn.Dropout(0.5))
            up_layers.append(norm_layer(outer_nc))

        model = down_layers
        if submodule is not None:
            model.append(submodule)
        model.extend(up_layers)
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)

"""Translation module networks (paper Sec. "Translation Module", Table 5).

* Generators G: real PX -> SimPX and H: SimPX -> real PX. The paper: "a UNet
  architecture consisting of four layers, with feature dimensions 64, 128,
  256 and 512". The same UNET class as the image encoder is used, with one
  output channel and tanh (CycleGAN works on images scaled to [-1, 1]).
* Discriminators D_X, D_Y: the 70 x 70 PatchGAN of CycleGAN (Zhu et al. 2017),
  the base model the paper names.
* Teeth segmentation model S: a UNet with a sigmoid output, trained on a
  public PX dataset with teeth masks; frozen while training the translation.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from ..models.image_encoder import UNET

__all__ = ["UNetGenerator", "PatchDiscriminator", "TeethSegmenter", "ImagePool", "init_weights"]


def _instance_norm(module: nn.Module) -> nn.Module:
    """Swap BatchNorm2d for InstanceNorm2d (CycleGAN's default normalisation).

    With batch size 1 and three different inputs per step (real PX, generated
    PX, SimPX for the identity loss), BatchNorm running statistics would mix
    domains and differ from what the generator sees in training.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(child.num_features, affine=False,
                                                    track_running_stats=False))
        else:
            _instance_norm(child)
    return module


class UNetGenerator(nn.Module):
    def __init__(self, features=(64, 128, 256, 512)):
        super().__init__()
        self.net = _instance_norm(UNET(in_channels=1, out_channels=1, features=list(features)))

    def forward(self, x):                                     
        return torch.tanh(self.net(x))


class PatchDiscriminator(nn.Module):
    """CycleGAN 70x70 PatchGAN: C64-C128-C256-C512, InstanceNorm, LeakyReLU(0.2)."""

    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1, bias=True),
                       nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1, bias=True),
                   nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TeethSegmenter(nn.Module):
    """UNet teeth segmentation for PX images in [0, 1]; returns the sigmoid map."""

    def __init__(self, features=(64, 128, 256, 512)):
        super().__init__()
        self.net = UNET(in_channels=1, out_channels=1, features=list(features))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class ImagePool:
    """History of generated images for the discriminators (CycleGAN, size 50)."""

    def __init__(self, size: int = 50):
        self.size, self.images = size, []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                j = random.randrange(self.size)
                out.append(self.images[j].clone())
                self.images[j] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def init_weights(net: nn.Module, gain: float = 0.02):
    """CycleGAN's normal(0, 0.02) initialisation."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or (isinstance(m, nn.InstanceNorm2d) and m.affine):
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.zeros_(m.bias)
    return net

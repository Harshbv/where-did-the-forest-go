"""The U-Net and its input preparation, shared by every script that runs the model.

Must match the Kaggle notebook exactly, or the saved weights won't fit / will be fed wrong inputs.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ROOT

MODEL_DIR = ROOT / "models"


class ConvBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_ch=5, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.down.append(ConvBlock(c, w)); c = w
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(ConvBlock(w * 2, w)); c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x).squeeze(1)


def load_model(version="v1"):
    """Trained U-Net (on CPU, in eval mode) plus the train mean/std it expects.
    v1: 4 bands (models/), v2: 6 bands with SWIR (models/v2/)."""
    folder = MODEL_DIR if version == "v1" else MODEL_DIR / version
    stats = json.load(open(folder / "results.json"))
    model = UNet(in_ch=len(stats["mean"]))
    model.load_state_dict(torch.load(folder / "best_unet.pt", map_location="cpu"))
    model.eval()
    return model, np.array(stats["mean"], "float32"), np.array(stats["std"], "float32")


def index(a, b):
    return (a - b) / (a + b + 1e-6)


def to_features(X, mean, std):
    """Raw bands uint16 -> standardised features.
    (N, 4, H, W) -> (N, 5, H, W): blue, green, red, nir + NDVI
    (N, 6, H, W) -> (N, 9, H, W): + swir1, swir2 + NDVI, NDMI (leaf moisture), NBR (burn)"""
    X = X.astype("float32") / 10000.0
    red, nir = X[:, 2], X[:, 3]
    extra = [index(nir, red)]
    if X.shape[1] == 6:
        extra += [index(nir, X[:, 4]), index(nir, X[:, 5])]
    feats = np.concatenate([X, np.stack(extra, axis=1)], axis=1)
    return (feats - mean[None, :, None, None]) / std[None, :, None, None]

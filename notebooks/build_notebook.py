"""Writes train_unet.ipynb from the cells below (keeps the notebook readable in git).

Run:  .venv/Scripts/python notebooks/build_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("md", """# Forest segmentation with a U-Net (Sentinel-2, Rondonia)
Input: 256x256 tiles with 4 bands (blue, green, red, near-infrared) + NDVI.
Output: a forest probability for every pixel.
Baseline to beat: NDVI threshold, **test IoU 0.853**.

**Before running:** Settings → Accelerator → **GPU T4 x2** (or P100)."""),

("code", """import glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42); np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

# Find our uploaded tiles wherever Kaggle mounted the dataset
paths = {s: glob.glob(f"/kaggle/input/**/{s}.npz", recursive=True)[0] for s in ["train", "val", "test"]}
data = {s: np.load(p) for s, p in paths.items()}
for s, d in data.items():
    print(s, d["X"].shape, d["y"].shape)"""),

("md", """## 1. Inputs: scale the bands
Raw values are reflectance x 10000. We add NDVI as a 5th channel, then standardise each
channel (subtract mean, divide by std) using **train statistics only**, so nothing about
the test area leaks into training."""),

("code", """def to_features(X):
    X = X.astype("float32") / 10000.0                      # reflectance 0..1
    red, nir = X[:, 2], X[:, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.concatenate([X, ndvi[:, None]], axis=1)      # N x 5 x 256 x 256

feats = {s: to_features(d["X"]) for s, d in data.items()}
labels = {s: d["y"] for s, d in data.items()}

valid_train = labels["train"] != 255
mean = np.array([feats["train"][:, c][valid_train].mean() for c in range(5)], dtype="float32")
std = np.array([feats["train"][:, c][valid_train].std() for c in range(5)], dtype="float32")
for s in feats:
    feats[s] = (feats[s] - mean[None, :, None, None]) / std[None, :, None, None]
print("channel means:", mean.round(3), "stds:", std.round(3))"""),

("md", """## 2. Dataset with augmentation
- **Flips/rotations:** satellite images have no "up", so a forest rotated 90 degrees is
  still a forest. 332 tiles effectively become 332 x 8 variations.
- **Brightness jitter:** each channel gets a small random scale and shift. Images from
  different years never have identical brightness (haze, smoke, sensor processing), and
  2026 pasture is greener than 2021 pasture. Jitter forces the model to rely on shape and
  texture, not exact values, so it transfers better to 2026 (fighting *domain shift*)."""),

("code", """class Tiles(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.augment:
            k = np.random.randint(4)                         # rotate 0/90/180/270
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:                       # mirror
                x, y = x[:, :, ::-1], y[:, ::-1]
            scale = np.random.uniform(0.9, 1.1, size=(len(x), 1, 1)).astype("float32")
            shift = np.random.normal(0, 0.15, size=(len(x), 1, 1)).astype("float32")
            x = x * scale + shift                            # brightness jitter (standardised units)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).long()

loaders = {
    "train": DataLoader(Tiles(feats["train"], labels["train"], augment=True), batch_size=16, shuffle=True, num_workers=2),
    "val": DataLoader(Tiles(feats["val"], labels["val"]), batch_size=16),
    "test": DataLoader(Tiles(feats["test"], labels["test"]), batch_size=16),
}"""),

("md", """## 3. The U-Net, written from scratch
- **Encoder (left side of the U):** conv blocks + max-pooling. The image shrinks
  256 → 128 → 64 → 32 → 16 while channels grow 32 → 512. Each level sees a wider area:
  first edges, then textures (rough canopy vs smooth pasture), then whole field shapes.
- **Decoder (right side):** upsamples back to 256x256 to say *where* things are.
- **Skip connections:** copy each encoder level across to the decoder, so the sharp
  detail lost during shrinking (exact field edges) comes back."""),

("code", """class ConvBlock(nn.Sequential):
    \"\"\"Two 3x3 convolutions, each followed by batch norm and ReLU.\"\"\"
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
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))   # double the size
            self.dec.append(ConvBlock(w * 2, w)); c = w            # *2: skip is concatenated
        self.head = nn.Conv2d(c, 1, 1)                             # 1 channel = forest score

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)                             # halve the size
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x).squeeze(1)                             # logits, N x 256 x 256

model = UNet().to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")"""),

("md", """## 4. Loss and metrics
- **BCE (binary cross-entropy):** per-pixel "how wrong was the probability?"
- **Dice loss:** directly rewards overlap between predicted and true forest areas,
  closer to the IoU we report. We use both.
- Pixels labelled 255 (no data) are ignored by both."""),

("code", """def loss_fn(logits, y):
    valid = y != 255
    target = (y == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
    p = torch.sigmoid(logits)[valid]
    dice = 1 - (2 * (p * target[valid]).sum() + 1) / (p.sum() + target[valid].sum() + 1)
    return bce + dice

@torch.no_grad()
def evaluate(loader, threshold=0.5):
    model.eval()
    tp = fp = fn = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, enabled=device == "cuda"):
            pred = torch.sigmoid(model(x)) > threshold
        valid = y != 255
        truth = (y == 1)
        tp += (pred & truth & valid).sum().item()
        fp += (pred & ~truth & valid).sum().item()
        fn += (~pred & truth & valid).sum().item()
        correct += ((pred == truth) & valid).sum().item()
        total += valid.sum().item()
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "acc": correct / max(total, 1)}"""),

("md", """## 5. Train
Adam optimiser, learning rate decays smoothly (cosine) over 40 epochs.
After every epoch we check **validation IoU** and keep the best weights, so a later,
over-fitted epoch can't overwrite a better model."""),

("code", """EPOCHS = 40
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
history, best_iou = [], -1.0

for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); running = 0.0
    for x, y in loaders["train"]:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device, enabled=device == "cuda"):   # mixed precision = faster on GPU
            loss = loss_fn(model(x).float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        running += loss.item() * len(x)
    sched.step()
    val = evaluate(loaders["val"])
    history.append({"epoch": epoch, "loss": running / len(loaders["train"].dataset), **val})
    flag = ""
    if val["iou"] > best_iou:
        best_iou = val["iou"]; torch.save(model.state_dict(), "best_unet.pt"); flag = "  <- best"
    print(f"epoch {epoch:2d}  loss {history[-1]['loss']:.4f}  val IoU {val['iou']:.4f}  "
          f"F1 {val['f1']:.4f}  ({time.time() - t0:.0f}s){flag}")"""),

("md", "## 6. Final exam: the test area (never seen during training)"),

("code", """model.load_state_dict(torch.load("best_unet.pt"))
test = evaluate(loaders["test"])
print(f"U-Net TEST  IoU {test['iou']:.4f}  F1 {test['f1']:.4f}  acc {test['acc']:.4f}")
print(f"NDVI baseline TEST IoU 0.853  ->  improvement {test['iou'] - 0.853:+.4f}")

import json
json.dump({"test": test, "history": history, "mean": mean.tolist(), "std": std.tolist()},
          open("results.json", "w"), indent=1)"""),

("code", """import matplotlib.pyplot as plt
h = history
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([r["loss"] for r in h]); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
ax[1].plot([r["iou"] for r in h], label="U-Net val IoU"); ax[1].axhline(0.853, ls="--", c="gray", label="NDVI baseline (test)")
ax[1].set_title("validation IoU"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=100); plt.show()"""),

("md", "## 7. Look at predictions: image, label, prediction, where it was wrong"),

("code", """model.eval()
idx = np.linspace(0, len(feats["test"]) - 1, 4).astype(int)
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for row, i in enumerate(idx):
    raw = data["test"]["X"][i].astype("float32")
    false_color = np.clip(np.stack([raw[3] / 4500, raw[2] / 3000, raw[1] / 3000], -1), 0, 1)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(feats["test"][i:i + 1]).to(device))).cpu().numpy()[0]
    y = labels["test"][i]
    wrong = np.where(y == 255, np.nan, (prob > 0.5) != (y == 1))
    for a, img, title, cmap in zip(axes[row], [false_color, y == 1, prob, wrong],
                                   ["false color", "label (WorldCover)", "U-Net forest probability", "errors (yellow)"],
                                   [None, "Greens", "Greens", "viridis"]):
        a.imshow(img, cmap=cmap, vmin=0, vmax=1); a.set_title(title); a.axis("off")
plt.tight_layout(); plt.savefig("test_predictions.png", dpi=90); plt.show()"""),

("md", """## 8. Download
Right panel → Output → download `best_unet.pt`, `results.json`, `training_curves.png`,
`test_predictions.png`, and put them in the project's `models/` folder."""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("train_unet.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(out)

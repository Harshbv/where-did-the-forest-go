"""v3: same U-Net and settings as v1 (4 bands + NDVI), trained on MapBiomas labels.

Runs as a Kaggle script kernel; outputs land in /kaggle/working.
"""
import glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.chdir("/kaggle/working")
torch.manual_seed(42); np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

paths = {s: glob.glob(f"/kaggle/input/**/{s}.npz", recursive=True)[0] for s in ["train", "val", "test"]}
data = {s: np.load(p) for s, p in paths.items()}
for s, d in data.items():
    print(s, d["X"].shape, d["y"].shape)


def to_features(X):
    X = X.astype("float32") / 10000.0
    red, nir = X[:, 2], X[:, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.concatenate([X, ndvi[:, None]], axis=1)


feats = {s: to_features(d["X"]) for s, d in data.items()}
labels = {s: d["y"] for s, d in data.items()}

# NDVI baseline with these labels: best threshold on train, scored on test.
def iou_of(pred, y):
    v = y != 255; t = y == 1
    return (pred & t & v).sum() / max(((pred | t) & v).sum(), 1)

best_t = max(np.arange(0.3, 0.9, 0.01), key=lambda t: iou_of(feats["train"][:, 4] > t, labels["train"]))
ndvi_test = iou_of(feats["test"][:, 4] > best_t, labels["test"])
print(f"NDVI baseline: threshold {best_t:.2f} (chosen on train), test IoU {ndvi_test:.4f}")

valid_train = labels["train"] != 255
mean = np.array([feats["train"][:, c][valid_train].mean() for c in range(5)], dtype="float32")
std = np.array([feats["train"][:, c][valid_train].std() for c in range(5)], dtype="float32")
for s in feats:
    feats[s] = (feats[s] - mean[None, :, None, None]) / std[None, :, None, None]


class Tiles(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            scale = np.random.uniform(0.9, 1.1, size=(len(x), 1, 1)).astype("float32")
            shift = np.random.normal(0, 0.15, size=(len(x), 1, 1)).astype("float32")
            x = x * scale + shift
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).long()


loaders = {
    "train": DataLoader(Tiles(feats["train"], labels["train"], augment=True), batch_size=16, shuffle=True, num_workers=2),
    "val": DataLoader(Tiles(feats["val"], labels["val"]), batch_size=16),
    "test": DataLoader(Tiles(feats["test"], labels["test"]), batch_size=16),
}


class ConvBlock(nn.Sequential):
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


def loss_fn(logits, y):
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
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "acc": correct / max(total, 1)}


model = UNet().to(device)
EPOCHS = 40
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
history, best_iou = [], -1.0

for epoch in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); running = 0.0
    for x, y in loaders["train"]:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device, enabled=device == "cuda"):
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
          f"F1 {val['f1']:.4f}  ({time.time() - t0:.0f}s){flag}", flush=True)

model.load_state_dict(torch.load("best_unet.pt"))
test = evaluate(loaders["test"])
print(f"v3 (MapBiomas labels) TEST  IoU {test['iou']:.4f}  F1 {test['f1']:.4f}  acc {test['acc']:.4f}"
      f"   NDVI baseline {ndvi_test:.4f}")
json.dump({"test": test, "ndvi_baseline": {"threshold": float(best_t), "test_iou": float(ndvi_test)},
           "history": history, "mean": mean.tolist(), "std": std.tolist(),
           "labels": "MapBiomas Collection 9, 2021"}, open("results.json", "w"), indent=1)
print("saved:", os.listdir("."))

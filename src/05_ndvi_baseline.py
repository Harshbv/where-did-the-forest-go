"""Phase 1d: the simplest possible forest detector, NDVI > threshold.

This is the score the U-Net has to beat. The threshold is chosen on TRAIN tiles
and only reported on TEST tiles, the same rules the U-Net will follow.
Run:  .venv/Scripts/python src/05_ndvi_baseline.py
"""
import numpy as np

from config import DATA_DIR


def ndvi_and_labels(split):
    d = np.load(DATA_DIR / "tiles" / f"{split}.npz")
    red, nir = d["X"][:, 2].astype("float32"), d["X"][:, 3].astype("float32")
    ndvi = (nir - red) / (nir + red + 1e-6)
    valid = d["y"] != 255
    return ndvi[valid], d["y"][valid].astype(bool)


def scores(pred, truth):
    tp = (pred & truth).sum()
    fp = (pred & ~truth).sum()
    fn = (~pred & truth).sum()
    iou = tp / (tp + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    acc = (pred == truth).mean()
    return iou, f1, acc


ndvi_tr, y_tr = ndvi_and_labels("train")
thresholds = np.arange(0.50, 0.95, 0.01)
best_t = max(thresholds, key=lambda t: scores(ndvi_tr > t, y_tr)[0])
print(f"Best NDVI threshold on train: {best_t:.2f}")

ndvi_te, y_te = ndvi_and_labels("test")
iou, f1, acc = scores(ndvi_te > best_t, y_te)
print(f"TEST  ->  forest IoU {iou:.3f}   F1 {f1:.3f}   pixel accuracy {acc:.3f}")
print(f"(Always guessing 'not forest' would score accuracy {1 - y_te.mean():.3f}, "
      f"which is why we report IoU, not accuracy.)")

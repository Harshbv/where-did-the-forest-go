"""Shows what's inside the tile dataset we upload to Kaggle."""
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, OUT_DIR

d = np.load(DATA_DIR / "tiles" / "train.npz")
X, y = d["X"], d["y"]
for split in ["train", "val", "test"]:
    f = DATA_DIR / "tiles" / f"{split}.npz"
    s = np.load(f)
    print(f"{f.name:<10} {f.stat().st_size / 1e6:6.1f} MB   X {s['X'].shape} {s['X'].dtype}   y {s['y'].shape} {s['y'].dtype}")

picks = np.random.default_rng(0).choice(len(X), 5, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
for col, i in enumerate(picks):
    nir, red, green = X[i, 3], X[i, 2], X[i, 1]
    img = np.clip(np.dstack([nir / 4500, red / 3000, green / 3000]), 0, 1)
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"tile #{i}  X: image (false color)", fontsize=10)
    axes[1, col].imshow(np.ma.masked_equal(y[i], 255), cmap="Greens", vmin=0, vmax=1)
    axes[1, col].set_title(f"y: label ({(y[i] == 1).mean():.0%} forest)", fontsize=10)
    for a in axes[:, col]:
        a.axis("off")
fig.suptitle("What one training example looks like: input image (top) + answer (bottom)\n"
             "Each tile = 256 x 256 pixels = 2.56 x 2.56 km", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_dataset_preview.png", dpi=100)

"""Phase 1c: cut the big image into 256x256 tiles and split them by location.

Run:  .venv/Scripts/python src/04_tiles.py
Makes: data/tiles/{train,val,test}.npz   (X: N x 4 x 256 x 256 uint16, y: N x 256 x 256 uint8)
       outputs/04_split_map.png
"""
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from matplotlib.patches import Patch

from config import DATA_DIR, OUT_DIR

TILE = 256
BLOCK = 4            # a "block" is 4x4 tiles (about 10 x 10 km)
MAX_NODATA = 0.10    # skip tiles that are more than 10% empty
SEED = 42

image = rioxarray.open_rasterio(DATA_DIR / "composite_before.tif").values  # (4, H, W)
labels = rioxarray.open_rasterio(DATA_DIR / "labels_2021.tif").values[0]   # (H, W)
n_rows, n_cols = labels.shape[0] // TILE, labels.shape[1] // TILE

# Spatial split. Neighboring tiles look almost identical, so a random per-tile
# split leaks: the model "memorizes" a field in train and meets its twin in test.
# Instead we assign whole 10 km blocks to train / val / test.
n_brows, n_bcols = -(-n_rows // BLOCK), -(-n_cols // BLOCK)
n_blocks = n_brows * n_bcols
n_val = n_test = round(0.15 * n_blocks)
# Shuffle the blocks, then deal out exact counts (random choice per block can
# come out lopsided when there are only a few dozen blocks).
order = np.random.default_rng(SEED).permutation(n_blocks)
block_split = np.empty(n_blocks, dtype=object)
block_split[order[:n_test]] = "test"
block_split[order[n_test:n_test + n_val]] = "val"
block_split[order[n_test + n_val:]] = "train"
block_split = block_split.reshape(n_brows, n_bcols)

tiles = {"train": ([], []), "val": ([], []), "test": ([], [])}
split_map = np.full((n_rows, n_cols), -1)
for r in range(n_rows):
    for c in range(n_cols):
        ys, xs = slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE)
        y = labels[ys, xs]
        if (y == 255).mean() > MAX_NODATA:
            continue
        split = block_split[r // BLOCK, c // BLOCK]
        tiles[split][0].append(image[:, ys, xs])
        tiles[split][1].append(y)
        split_map[r, c] = ["train", "val", "test"].index(split)

out_dir = DATA_DIR / "tiles"
out_dir.mkdir(exist_ok=True)
for split, (X, y) in tiles.items():
    X, y = np.stack(X), np.stack(y)
    np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y)
    forest = (y == 1).sum() / (y != 255).sum()
    print(f"{split:<5}: {len(X):4d} tiles   forest pixels {forest:.1%}")

colors = np.array([[0.85, 0.85, 0.85], [0.2, 0.5, 0.9], [1.0, 0.6, 0.1], [0.85, 0.2, 0.3]])
plt.figure(figsize=(7, 7))
plt.imshow(colors[split_map + 1])
plt.legend(handles=[Patch(color=colors[i + 1], label=n) for i, n in enumerate(["train", "val", "test"])]
           + [Patch(color=colors[0], label="skipped")], loc="upper right")
plt.title("Which tiles go where (each square = 2.56 x 2.56 km)")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_split_map.png", dpi=100)
print(f"Saved {OUT_DIR / '04_split_map.png'}")

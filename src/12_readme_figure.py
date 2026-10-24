"""README figure: zoom into the biggest clearings, natural color, 2021 vs 2026 vs detected.

Run:  .venv/Scripts/python src/12_readme_figure.py
Makes: outputs/12_before_after.png
"""
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from scipy import ndimage

from config import DATA_DIR, OUT_DIR

HALF = 400    # 800 x 800 px crop = 8 x 8 km

change = rioxarray.open_rasterio(DATA_DIR / "change_unet.tif").values[0]
loss = change == 1

# Centre the crop where cleared area is densest (smoothed loss map, 4 km box).
density = ndimage.uniform_filter(loss.astype("float32")[::8, ::8], size=50)
r, c = np.unravel_index(density.argmax(), density.shape)
r, c = np.clip(r * 8, HALF, loss.shape[0] - HALF), np.clip(c * 8, HALF, loss.shape[1] - HALF)
win = (slice(r - HALF, r + HALF), slice(c - HALF, c + HALF))


def natural_color(name):
    x = rioxarray.open_rasterio(DATA_DIR / f"{name}.tif").values[:3, win[0], win[1]].astype("float32")
    return np.clip(np.dstack([x[2], x[1], x[0]]) / 1800, 0, 1) ** (1 / 1.6)


before, after = natural_color("composite_before"), natural_color("composite_after_norm")
overlay = after.copy()
overlay[loss[win]] = 0.35 * overlay[loss[win]] + 0.65 * np.array([1.0, 0.1, 0.1])

fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
titles = ["July 2021", "July 2026", f"Detected clearing (red): {loss[win].sum() * 0.01:,.0f} ha in this 8 x 8 km view"]
for a, img, t in zip(ax, [before, after, overlay], titles):
    a.imshow(img)
    a.set_title(t, fontsize=14)
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "12_before_after.png", dpi=80)
print(f"Saved {OUT_DIR / '12_before_after.png'}  (crop centre row {r}, col {c})")

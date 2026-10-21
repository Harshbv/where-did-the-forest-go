"""Phase 1b: get the "answer key" (forest / not forest) for every pixel.

Uses ESA WorldCover 2021, a global 10 m land-cover map. Class 10 = tree cover.
Run:  .venv/Scripts/python src/03_labels.py
Makes: data/labels_2021.tif (1 = forest, 0 = not forest, 255 = no data)
       outputs/03_labels_check.png
"""
import matplotlib.pyplot as plt
import numpy as np
import odc.geo.xr  # noqa: F401  (adds .odc.geobox to xarray objects)
import planetary_computer
import pystac_client
import rioxarray
from odc.stac import load

from config import BBOX, DATA_DIR, OUT_DIR, STAC_URL

TREE_COVER = 10

# Use the 2021 composite as the "template" grid: same CRS, same 10 m pixels,
# same corners. Labels MUST line up pixel-for-pixel with the image, otherwise
# the model is taught that pixel (i, j) is forest when it's really the field next door.
comp = rioxarray.open_rasterio(DATA_DIR / "composite_before.tif")
grid = comp.odc.geobox

catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
items = list(catalog.search(collections=["esa-worldcover"], bbox=BBOX,
                            datetime="2021-01-01/2021-12-31").items())
print(f"WorldCover 2021 tiles touching our box: {len(items)}")

# resampling="nearest": class numbers are categories, never average them
# (the mean of "tree" 10 and "cropland" 40 is 25, which is meaningless).
wc = load(items, bands=["map"], geobox=grid, resampling="nearest").isel(time=0)["map"].values

labels = (wc == TREE_COVER).astype("uint8")
labels[wc == 0] = 255                                  # WorldCover has no data here
labels[(comp.values == 0).all(axis=0)] = 255           # our image has no data here

out = comp.isel(band=0).copy(data=labels).rio.write_nodata(255)
out.rio.to_raster(DATA_DIR / "labels_2021.tif", compress="deflate")

valid = labels != 255
print(f"Saved {DATA_DIR / 'labels_2021.tif'}  shape {labels.shape}")
print(f"Forest share (WorldCover): {labels[valid].mean():.1%}")
classes, counts = np.unique(wc, return_counts=True)
names = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built-up",
         60: "bare", 80: "water", 90: "wetland"}
print("All WorldCover classes here:",
      ", ".join(f"{names.get(c, c)} {n / wc.size:.1%}" for c, n in zip(classes, counts)))

# Visual check: image and label side by side (every 4th pixel, to keep it light).
s = slice(None, None, 4)
nir, red, green = (comp.sel(band=b).values[s, s] for b in (4, 3, 2))
false_color = np.clip(np.dstack([nir, red, green]) / np.array([4500, 3000, 3000]), 0, 1)
fig, ax = plt.subplots(1, 2, figsize=(16, 8.5))
ax[0].imshow(false_color)
ax[0].set_title("2021 cloud-free composite (false color)", fontsize=13)
ax[1].imshow(np.ma.masked_equal(labels[s, s], 255), cmap="Greens", vmin=0, vmax=1)
ax[1].set_title("WorldCover label: green = forest, white = not forest", fontsize=13)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_labels_check.png", dpi=100)
print(f"Saved {OUT_DIR / '03_labels_check.png'}")

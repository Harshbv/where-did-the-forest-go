"""Phase 3c: compare the 2021 and 2026 forest maps -> where was forest cleared?

Rules (each one removes a kind of false alarm):
  1. Confident change only: forest probability >= 60% in 2021 AND <= 40% in 2026.
     Pixels near 50% flip back and forth from noise alone.
  2. Morphological opening: erase strips 1-2 pixels wide. Forest edges shift by a pixel
     between dates (slightly different viewing geometry), which creates thin false
     "loss" slivers along every border.
  3. Minimum mapping unit (MMU): drop patches smaller than 0.5 ha (50 pixels).

Run:  .venv/Scripts/python src/09_change.py          (v1)
      .venv/Scripts/python src/09_change.py v2       (v2: SWIR model; writes *_v2 files)
Makes: data/change_unet.tif, data/change_ndvi.tif (0 none, 1 loss, 2 gain, 255 no data)
       outputs/loss_patches.geojson, outputs/09_change_overview.png
"""
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from rasterio.features import shapes
from scipy import ndimage

from config import DATA_DIR, OUT_DIR

V = "_v2" if "v2" in sys.argv[1:] else ""     # file name suffix
HIGH, LOW = 60, 40
MMU_PIXELS = 50           # 50 px x 100 m2 = 0.5 ha
PIXEL_HA = 0.01           # one 10 m x 10 m pixel = 100 m2 = 0.01 ha
NDVI_T = 0.61             # baseline threshold chosen on train tiles in Phase 1


def clean(mask):
    """Opening (remove thin slivers), then drop patches smaller than the MMU."""
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= MMU_PIXELS
    keep[0] = False
    return keep[lab]


def save(change, da, name):
    da.isel(band=0).copy(data=change).rio.write_nodata(255).rio.to_raster(
        DATA_DIR / f"change_{name}.tif", compress="deflate")


p_before = rioxarray.open_rasterio(DATA_DIR / f"forest_prob_before{V}.tif")
pb = p_before.values[0]
pa = rioxarray.open_rasterio(DATA_DIR / f"forest_prob_after{V}.tif").values[0]
valid = (pb != 255) & (pa != 255)

# ---- U-Net change ----
loss = clean(valid & (pb >= HIGH) & (pa <= LOW))
gain = clean(valid & (pb <= LOW) & (pa >= HIGH))
change = np.zeros(pb.shape, "uint8")
change[loss], change[gain], change[~valid] = 1, 2, 255
save(change, p_before, f"unet{V}")

forest21 = (pb > 50) & valid
forest26 = (pa > 50) & valid
print("U-Net forest map")
print(f"   forest 2021: {forest21.sum() * PIXEL_HA:,.0f} ha    forest 2026: {forest26.sum() * PIXEL_HA:,.0f} ha")
print(f"   CLEARED 2021->2026: {loss.sum() * PIXEL_HA:,.0f} ha  "
      f"({loss.sum() / forest21.sum():.1%} of 2021 forest)")
print(f"   regrowth: {gain.sum() * PIXEL_HA:,.0f} ha")

# ---- NDVI baseline change (same cleaning rules, for comparison) ----
def ndvi(path):
    x = rioxarray.open_rasterio(path).values.astype("float32")
    return (x[3] - x[2]) / (x[3] + x[2] + 1e-6)

nb, na = ndvi(DATA_DIR / "composite_before.tif"), ndvi(DATA_DIR / "composite_after_norm.tif")
loss_ndvi = clean(valid & (nb > NDVI_T) & (na <= NDVI_T))
gain_ndvi = clean(valid & (nb <= NDVI_T) & (na > NDVI_T))
change_ndvi = np.zeros(pb.shape, "uint8")
change_ndvi[loss_ndvi], change_ndvi[gain_ndvi], change_ndvi[~valid] = 1, 2, 255
save(change_ndvi, p_before, "ndvi")
print(f"NDVI baseline: cleared {loss_ndvi.sum() * PIXEL_HA:,.0f} ha, regrowth {gain_ndvi.sum() * PIXEL_HA:,.0f} ha")
del nb, na

# ---- Loss patches as polygons (for the map) ----
lab, n = ndimage.label(loss, structure=np.ones((3, 3)))
transform = p_before.rio.transform()
geoms = [{"type": "Feature", "geometry": g, "properties": {"patch_id": int(v)}}
         for g, v in shapes(lab.astype("int32"), mask=lab > 0, transform=transform)]
patches = gpd.GeoDataFrame.from_features(geoms, crs=p_before.rio.crs).dissolve("patch_id").reset_index()
patches["area_ha"] = (patches.area / 10_000).round(2)
idx = np.arange(1, n + 1)
patches = patches.set_index("patch_id")
patches["forest_prob_2021"] = ndimage.mean(pb, lab, idx).round(0)
patches["forest_prob_2026"] = ndimage.mean(pa, lab, idx).round(0)
patches = patches.reset_index().sort_values("area_ha", ascending=False)
patches["geometry"] = patches.simplify(10)        # 10 m tolerance: lighter web map
patches.to_crs("EPSG:4326").to_file(OUT_DIR / f"loss_patches{V}.geojson", driver="GeoJSON")
print(f"{len(patches)} cleared patches   largest {patches.area_ha.max():.0f} ha   "
      f"median {patches.area_ha.median():.1f} ha")
print("size classes:", ", ".join(
    f"{lo}-{hi} ha: {((patches.area_ha >= lo) & (patches.area_ha < hi)).sum()}"
    for lo, hi in [(0.5, 1), (1, 5), (5, 25), (25, 100), (100, 10_000)]))

# ---- Overview figure ----
s = slice(None, None, 4)

def false_color(path):
    x = rioxarray.open_rasterio(path).values[:, s, s].astype("float32")
    return np.clip(np.dstack([x[3] / 4500, x[2] / 3000, x[1] / 3000]), 0, 1)

fig, ax = plt.subplots(1, 3, figsize=(21, 7.5))
ax[0].imshow(false_color(DATA_DIR / "composite_before.tif")); ax[0].set_title("July-Aug 2021", fontsize=14)
ax[1].imshow(false_color(DATA_DIR / "composite_after_norm.tif")); ax[1].set_title("July-Aug 2026", fontsize=14)
overlay = np.full(change[s, s].shape + (3,), 0.93)
overlay[forest21[s, s] & ~loss[s, s]] = (0.55, 0.75, 0.55)
overlay[gain[s, s]] = (0.2, 0.45, 0.9)
overlay[loss[s, s]] = (0.85, 0.05, 0.05)
ax[2].imshow(overlay)
ax[2].set_title(f"RED = cleared ({loss.sum() * PIXEL_HA:,.0f} ha)   BLUE = regrowth   green = forest kept",
                fontsize=12)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / f"09_change_overview{V}.png", dpi=90)
print(f"Saved {OUT_DIR / f'09_change_overview{V}.png'}")

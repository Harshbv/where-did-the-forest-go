"""Phase 0: search the satellite catalog and look at our first image.

Run:  .venv/Scripts/python src/01_search.py
Makes: outputs/01_true_color.png, outputs/01_false_color.png, outputs/01_ndvi.png
"""
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load
from shapely.geometry import box, shape

from config import BBOX, COLLECTION, CRS, DATES, OUT_DIR, STAC_URL

# 1) Open the catalog. sign_inplace adds a temporary access token to every
#    file link, which Planetary Computer requires before you can download.
catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

# 2) Search: which scenes touch our box, in the dry season, with few clouds?
search = catalog.search(
    collections=[COLLECTION],
    bbox=BBOX,
    datetime=DATES["before"],
    query={"eo:cloud_cover": {"lt": 10}},
)
items = list(search.items())
print(f"Found {len(items)} scenes with <10% cloud\n")

# 3) A scene is one 110x110 km tile, so it may only partly cover our box.
#    Measure how much of our box each scene actually covers.
aoi = box(*BBOX)
rows = []
for it in items:
    coverage = shape(it.geometry).intersection(aoi).area / aoi.area
    rows.append((it, coverage))
    print(f"{it.id:<50} {it.datetime:%Y-%m-%d}  "
          f"cloud {it.properties['eo:cloud_cover']:5.2f}%  covers {coverage:5.0%}")

# 4) Best scene = covers (almost) all of our box, then fewest clouds.
best, cov = min(rows, key=lambda r: (r[1] < 0.95, r[0].properties["eo:cloud_cover"]))
print(f"\nUsing {best.id}  ({cov:.0%} coverage)")

# 5) Download only the pixels inside our box. resolution=20 is a quick preview
#    (Phase 1 will use the full 10 m). odc-stac reprojects to our UTM CRS.
ds = load([best], bands=["B02", "B03", "B04", "B08"], bbox=BBOX,
          crs=CRS, resolution=20).isel(time=0)
print(f"Image size: {ds.sizes['y']} x {ds.sizes['x']} pixels")

# Values are reflectance x 10000 (so 3000 means 30% of light reflected).
blue, green, red, nir = (ds[b].values.astype("float32") for b in ["B02", "B03", "B04", "B08"])


def stretch(*bands, max_val=3000):
    """Stack 3 bands into an RGB image and scale them to 0-1 for display."""
    return np.clip(np.dstack(bands) / max_val, 0, 1)


# True color: what your eyes would see.
plt.imsave(OUT_DIR / "01_true_color.png", stretch(red, green, blue))

# False color: near-infrared shown as red. Healthy forest glows bright red,
# cleared land looks cyan/brown. This is the classic remote-sensing view.
plt.imsave(OUT_DIR / "01_false_color.png", stretch(nir, red, green, max_val=4500))

# NDVI = (NIR - Red) / (NIR + Red). Ranges -1..1; dense forest is ~0.8+.
ndvi = (nir - red) / (nir + red + 1e-6)
plt.imsave(OUT_DIR / "01_ndvi.png", ndvi, cmap="RdYlGn", vmin=0, vmax=1)

print(f"Mean NDVI: {np.nanmean(ndvi):.2f}   "
      f"pixels with NDVI > 0.7 (likely forest): {(ndvi > 0.7).mean():.0%}")
print(f"Saved images to {OUT_DIR}")

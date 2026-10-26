"""Phase 6a: better training labels from MapBiomas (Brazil's yearly land cover maps).

Why: WorldCover's "tree cover" includes plantations and bushy young regrowth, so a model
trained on it copies that habit (our hand check found it was the biggest error source,
and better input bands didn't fix it). MapBiomas separates natural forest from
plantations, pasture, crops and "mosaic of uses", every year since 1985.

Our forest definition (decided BEFORE seeing any results):
    1   forest  = natural forest in 2021 (classes 3 forest formation, 6 floodable forest)
                  that has NOT been in human use in the last 10 years
    255 ignore  = natural forest in 2021 that WAS in human use within 10 years (young
                  regrowth: mixed, ambiguous), and savanna / wetland / not observed
    0   not forest = everything else (pasture, crops, plantations, mosaic, urban, water...)

Young regrowth is found from the yearly history (the same logic as MapBiomas' own
secondary vegetation layer, which is only published on Earth Engine).

Run:  .venv/Scripts/python src/03b_labels_mapbiomas.py
Makes: data/labels_2021_mapbiomas.tif (on the exact composite grid), outputs/03b_labels_compare.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from config import BBOX, DATA_DIR, OUT_DIR

URL = ("/vsicurl/https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
       "collection_9/lclu/coverage/brasil_coverage_{year}.tif")
YEARS = range(2011, 2022)          # the last 10 years before 2021, plus 2021 itself
NATURAL_FOREST = [3, 6]
AMBIGUOUS = [4, 11, 12, 27, 0]     # savanna, wetland, grassland, not observed, no data
HUMAN_USE = [9, 15, 18, 19, 20, 21, 24, 30, 31, 35, 36, 39, 40, 41, 46, 47, 48, 62]
MARGIN = 0.01                      # read a bit beyond our box so reprojection has no gaps

bounds = (BBOX[0] - MARGIN, BBOX[1] - MARGIN, BBOX[2] + MARGIN, BBOX[3] + MARGIN)


def read(year, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            with rasterio.open(URL.format(year=year)) as src:
                win = from_bounds(*bounds, src.transform)
                return src.read(1, window=win), src.window_transform(win)
        except rasterio.errors.RasterioIOError as e:
            print(f"   retry {attempt} for {year}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not read MapBiomas {year}")


stack = {}
for y in YEARS:
    stack[y], transform = read(y)
    print(f"read {y}: {stack[y].shape}")

lc21 = stack[2021]
used_recently = np.zeros(lc21.shape, bool)
for y in YEARS:
    if y < 2021:
        used_recently |= np.isin(stack[y], HUMAN_USE)

forest = np.isin(lc21, NATURAL_FOREST)
labels = np.zeros(lc21.shape, "uint8")
labels[forest & ~used_recently] = 1
labels[forest & used_recently] = 255            # young regrowth
labels[np.isin(lc21, AMBIGUOUS)] = 255

# Put it on our exact 10 m grid. Nearest: these are categories, never average them.
lab = rioxarray.open_rasterio(DATA_DIR / "labels_2021.tif").isel(band=0)  # WorldCover labels: our grid
raw = xr.DataArray(labels, dims=("y", "x")).rio.write_crs("EPSG:4326").rio.write_transform(transform)
raw = raw.assign_coords(
    x=transform.c + transform.a * (np.arange(labels.shape[1]) + 0.5),
    y=transform.f + transform.e * (np.arange(labels.shape[0]) + 0.5)).rio.write_nodata(255)
mb = raw.rio.reproject_match(lab, resampling=Resampling.nearest)
mb.rio.write_nodata(255).rio.to_raster(DATA_DIR / "labels_2021_mapbiomas.tif", compress="deflate")

wc = lab.values
m = mb.values
both = (wc != 255) & (m != 255)
print(f"\nMapBiomas labels: forest {np.mean(m[m != 255] == 1):.1%}, "
      f"ignored {np.mean(m == 255):.1%} (young regrowth {np.mean(raw.values == 255):.1%} incl. ambiguous)")
print(f"WorldCover forest {np.mean(wc[wc != 255] == 1):.1%}")
print(f"Where both have a label: agree {np.mean(wc[both] == m[both]):.1%}   "
      f"WorldCover forest but MapBiomas not: {np.mean((wc == 1) & (m == 0) & both) / np.mean(both):.1%} of area   "
      f"the reverse: {np.mean((wc == 0) & (m == 1) & both) / np.mean(both):.1%}")

# Picture: where the two answer keys disagree.
s = slice(None, None, 4)
before = rioxarray.open_rasterio(DATA_DIR / "composite_before.tif").values[:3, s, s].astype("float32")
rgb = np.clip(np.dstack([before[2], before[1], before[0]]) / 1800, 0, 1) ** (1 / 1.6)
diff = np.full(m[s, s].shape + (3,), 0.93)
diff[(wc[s, s] == 1) & (m[s, s] == 1)] = (0.3, 0.6, 0.3)
diff[(wc[s, s] == 1) & (m[s, s] == 0)] = (0.95, 0.6, 0.0)
diff[(wc[s, s] == 0) & (m[s, s] == 1)] = (0.2, 0.4, 0.9)
diff[m[s, s] == 255] = (0.6, 0.6, 0.6)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(rgb); ax[0].set_title("July-Aug 2021", fontsize=14)
ax[1].imshow(diff)
ax[1].set_title("green both say forest, ORANGE only WorldCover, BLUE only MapBiomas, grey ignored", fontsize=11)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "03b_labels_compare.png", dpi=80)
print(f"Saved {DATA_DIR / 'labels_2021_mapbiomas.tif'} and {OUT_DIR / '03b_labels_compare.png'}")

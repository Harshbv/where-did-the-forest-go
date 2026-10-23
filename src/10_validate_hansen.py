"""Phase 3d: check our cleared areas against Hansen Global Forest Change (v1.13).

Hansen et al. map forest loss worldwide every year from Landsat (30 m). "lossyear"
says in which year a pixel lost its trees: 1 = 2001 ... 25 = 2025, 0 = no loss.

Our window is Aug 2021 -> Aug 2026, which doesn't line up perfectly with Hansen years:
  - loss in 2022-2025 is definitely inside our window  -> "reference loss"
  - loss in 2021 may be before or after our Aug 2021 image -> ambiguous
  - Jan-Aug 2026 isn't in Hansen yet -> our real precision is a bit higher than measured

Run:  .venv/Scripts/python src/10_validate_hansen.py
"""
import numpy as np
import rioxarray
from rasterio.enums import Resampling

from config import BBOX, DATA_DIR

URL = ("https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/"
       "Hansen_GFC-2025-v1.13_lossyear_00N_070W.tif")
PIXEL_HA = 0.01

grid = rioxarray.open_rasterio(DATA_DIR / "change_unet.tif")
hansen_path = DATA_DIR / "hansen_lossyear.tif"
if not hansen_path.exists():
    # Read only our box out of the 40000 x 40000 px tile, over HTTP.
    h = rioxarray.open_rasterio(URL).rio.clip_box(*BBOX, crs="EPSG:4326")
    h.rio.to_raster(hansen_path, compress="deflate")
    print(f"downloaded Hansen subset {h.shape}")
# Put Hansen's 30 m pixels on our exact 10 m grid (nearest: years are categories).
lossyear = rioxarray.open_rasterio(hansen_path).rio.reproject_match(
    grid, resampling=Resampling.nearest).values[0]

ref_loss = (lossyear >= 22) & (lossyear <= 25)
any_recent = (lossyear >= 21)
print("Hansen loss in our area: " + ", ".join(
    f"20{y}: {(lossyear == y).sum() * PIXEL_HA:,.0f} ha" for y in range(21, 26)))


def report(name, change):
    valid = change != 255
    ours = (change == 1)
    # Precision: of what we flagged, how much did Hansen also see cleared (2021-2025)?
    precision = (ours & any_recent).sum() / ours.sum()
    # Recall: of Hansen's 2022-2025 clearing, how much did we find?
    recall = (ours & ref_loss & valid).sum() / (ref_loss & valid).sum()
    print(f"{name:<14} flagged {ours.sum() * PIXEL_HA:7,.0f} ha   "
          f"precision {precision:.1%}   recall {recall:.1%}")
    return precision, recall


print()
report("U-Net", grid.values[0])
report("NDVI baseline", rioxarray.open_rasterio(DATA_DIR / "change_ndvi.tif").values[0])
print("\nPixel-level agreement is a strict test: Hansen is 30 m vs our 10 m, so patch edges never "
      "line up exactly. Patch-level check below is fairer.")

# Patch-level: a patch counts as "confirmed" if at least 30% of it is Hansen loss.
from scipy import ndimage
for name, path in [("U-Net", "change_unet.tif"), ("NDVI baseline", "change_ndvi.tif")]:
    ours = rioxarray.open_rasterio(DATA_DIR / path).values[0] == 1
    lab, n = ndimage.label(ours, structure=np.ones((3, 3)))
    overlap = ndimage.mean(any_recent, lab, np.arange(1, n + 1))
    area = ndimage.sum(ours, lab, np.arange(1, n + 1)) * PIXEL_HA
    confirmed = overlap >= 0.3
    print(f"{name:<14} {n:5d} patches, {confirmed.mean():.0%} confirmed by Hansen   "
          f"(= {area[confirmed].sum() / area.sum():.0%} of flagged area)")

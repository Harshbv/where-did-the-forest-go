"""Phase 1a: build one cloud-free image per period (a "median composite").

Two stages:
  1. Download each scene's bands for our box into data/cache/ (with retries).
     Already-downloaded files are skipped, so a crash just means "run it again".
  2. Build the median composite from the local files (no network = no surprises).

Run:  .venv/Scripts/python src/02_composite.py            (both periods)
      .venv/Scripts/python src/02_composite.py after      (just one)
      .venv/Scripts/python src/02_composite.py swir       (SWIR bands, both periods)
Makes: data/composite_before.tif, data/composite_after.tif  (bands: blue, green, red, nir)
       data/composite_{before,after}_swir.tif               (bands: swir1, swir2)
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
from odc.stac import load
from shapely.geometry import box, shape

from config import BBOX, COLLECTION, CRS, DATA_DIR, DATES, STAC_URL

# SCL (scene classification) codes we throw away:
# 0 no data, 1 saturated, 3 cloud shadow, 8 cloud (medium), 9 cloud (high), 10 thin cirrus
BAD_SCL = [0, 1, 3, 8, 9, 10]
# Band sets: (asset names, output names, file suffix, resampling).
# SWIR (B11, B12) is measured at 20 m. Bilinear resampling to our 10 m grid gives
# smooth values instead of blocky 2 x 2 copies. It shows water in leaves and soil:
# moist forest canopy is dark, crops / scrub / bare soil are bright.
BAND_SETS = {
    "main": (["B02", "B03", "B04", "B08"], ["blue", "green", "red", "nir"], "", "nearest"),
    "swir": (["B11", "B12"], ["swir1", "swir2"], "_swir", "bilinear"),
}
MAX_SCENES = 8
CACHE = DATA_DIR / "cache"
CACHE.mkdir(exist_ok=True)

catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
aoi = box(*BBOX)


def find_scenes(period):
    """Fairly clear scenes that cover (almost) all of our box, clearest first."""
    items = catalog.search(collections=[COLLECTION], bbox=BBOX, datetime=period,
                           query={"eo:cloud_cover": {"lt": 40}}).items()
    full = [it for it in items if shape(it.geometry).intersection(aoi).area / aoi.area > 0.95]
    full.sort(key=lambda it: it.properties["eo:cloud_cover"])
    return full[:MAX_SCENES]


def fetch(item, band, resampling="nearest", attempts=6):
    """Download one band of one scene (clipped to our box) to a local GeoTIFF."""
    path = CACHE / f"{item.id}_{band}.tif"
    if path.exists():
        return path
    for attempt in range(1, attempts + 1):
        try:
            # SCL holds class codes, so it always uses nearest neighbour (the default).
            da = load([item], bands=[band], bbox=BBOX, crs=CRS, resolution=10,
                      resampling=resampling).isel(time=0)[band]
            tmp = path.with_suffix(".part.tif")
            da.rio.to_raster(tmp, tiled=True, compress="deflate")
            tmp.rename(path)  # only a complete file gets the real name
            return path
        except Exception as e:  # dropped connection, empty block, expired token...
            print(f"   retry {attempt}/{attempts} {item.id[:28]} {band}: {type(e).__name__}")
            time.sleep(5 * attempt)
            item = planetary_computer.sign(item)  # fresh access token
    raise RuntimeError(f"gave up on {item.id} {band}")


def build_composite(name, band_set="main"):
    asset_bands, band_names, suffix, resampling = BAND_SETS[band_set]
    items = sorted(find_scenes(DATES[name]), key=lambda it: it.datetime)
    print(f"\n[{name}] using {len(items)} scenes:")
    for it in items:
        print(f"   {it.datetime:%Y-%m-%d}  cloud {it.properties['eo:cloud_cover']:5.2f}%  "
              f"baseline {it.properties['s2:processing_baseline']}")

    # Stage 1: download, 4 files at a time.
    jobs = [(it, b, resampling) for it in items for b in asset_bands] + [(it, "SCL") for it in items]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    print(f"[{name}] all {len(jobs)} band files cached")

    # Stage 2: stack the local files into (time, y, x) arrays, read lazily in chunks.
    def stack(band):
        return xr.concat([rioxarray.open_rasterio(CACHE / f"{it.id}_{band}.tif",
                                                  chunks={"x": 1024, "y": 1024}).squeeze("band", drop=True)
                          for it in items], dim="time")

    # Gotcha: from Jan 2022 ESA added +1000 to every reflectance value
    # (processing baseline 04.00+). Without undoing that, 2026 looks brighter
    # than 2021 and we'd "detect" change that isn't real.
    offset = xr.DataArray(
        [1000.0 if float(it.properties["s2:processing_baseline"]) >= 4 else 0.0 for it in items],
        dims="time")

    good = ~stack("SCL").isin(BAD_SCL)          # True where the pixel is clear
    bands = []
    for b in asset_bands:
        raw = stack(b)
        refl = (raw.astype("float32") - offset).clip(min=0)
        refl = refl.where(good & (raw > 0))      # cloudy / empty pixels become NaN
        # Median over time ignores NaNs: each pixel gets its "typical" clear value.
        # Median (not mean) so a missed cloud or shadow can't drag the value.
        bands.append(refl.median("time", skipna=True))

    comp = xr.concat(bands, dim="band").assign_coords(band=band_names)
    comp = comp.fillna(0).round().astype("uint16").rio.write_crs(CRS).rio.write_nodata(0)
    out = DATA_DIR / f"composite_{name}{suffix}.tif"
    with dask.config.set(scheduler="threads", num_workers=4):
        comp.rio.to_raster(out, tiled=True, compress="deflate", lock=threading.Lock())
        n_clear = good.sum("time").compute().values
    print(f"[{name}] saved {out}  ({comp.sizes['y']} x {comp.sizes['x']} px)")
    print(f"[{name}] pixels with 0 clear looks: {(n_clear == 0).mean():.2%}   "
          f"median clear looks per pixel: {np.median(n_clear):.0f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    band_set = "swir" if "swir" in args else "main"
    for name in [a for a in args if a in DATES] or DATES:
        build_composite(name, band_set)

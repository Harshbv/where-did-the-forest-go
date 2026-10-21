"""Phase 1a: build one cloud-free image per period (a "median composite").

Run:  .venv/Scripts/python src/02_composite.py
Makes: data/composite_before.tif, data/composite_after.tif  (bands: blue, green, red, nir)
"""
import os
import threading

# Downloads are hundreds of small web requests; if one drops, retry it instead
# of crashing 30 minutes in. GDAL (the library under rasterio) reads these settings.
os.environ.update({
    "GDAL_HTTP_MAX_RETRY": "8",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
})

import dask
import numpy as np
import planetary_computer
import pystac_client
import xarray as xr
from odc.stac import load
from shapely.geometry import box, shape

from config import BBOX, COLLECTION, CRS, DATA_DIR, DATES, STAC_URL

# SCL (scene classification) codes we throw away:
# 0 no data, 1 saturated, 3 cloud shadow, 8 cloud (medium), 9 cloud (high), 10 thin cirrus
BAD_SCL = [0, 1, 3, 8, 9, 10]
REFLECTANCE_BANDS = ["B02", "B03", "B04", "B08"]
MAX_SCENES = 8

catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
aoi = box(*BBOX)


def find_scenes(period):
    """Fairly clear scenes that cover (almost) all of our box, clearest first."""
    items = catalog.search(collections=[COLLECTION], bbox=BBOX, datetime=period,
                           query={"eo:cloud_cover": {"lt": 40}}).items()
    full = [it for it in items if shape(it.geometry).intersection(aoi).area / aoi.area > 0.95]
    full.sort(key=lambda it: it.properties["eo:cloud_cover"])
    return full[:MAX_SCENES]


def build_composite(name):
    items = find_scenes(DATES[name])
    print(f"\n[{name}] using {len(items)} scenes:")
    for it in items:
        print(f"   {it.datetime:%Y-%m-%d}  cloud {it.properties['eo:cloud_cover']:5.2f}%  "
              f"baseline {it.properties['s2:processing_baseline']}")

    # odc-stac returns time steps in date order, so sort our list the same way.
    items.sort(key=lambda it: it.datetime)

    # chunks= makes this "lazy": nothing downloads yet, we just describe the work.
    # Dask later runs it in 1024x1024 pieces so it fits in a small amount of RAM.
    ds = load(items, bands=REFLECTANCE_BANDS + ["SCL"], bbox=BBOX, crs=CRS,
              resolution=10, chunks={"x": 1024, "y": 1024})
    assert ds.sizes["time"] == len(items), "expected one time step per scene"

    # Gotcha: from Jan 2022 ESA added +1000 to every reflectance value
    # (processing baseline 04.00+). Without undoing that, 2024 looks brighter
    # than 2021 and we'd "detect" change that isn't real.
    offset = xr.DataArray(
        [1000.0 if float(it.properties["s2:processing_baseline"]) >= 4 else 0.0 for it in items],
        dims="time", coords={"time": ds.time})

    good = ~ds["SCL"].isin(BAD_SCL)            # True where the pixel is clear
    bands = []
    for b in REFLECTANCE_BANDS:
        refl = (ds[b].astype("float32") - offset).clip(min=0)
        refl = refl.where(good)                 # cloudy pixels become NaN
        # Median over time ignores NaNs: each pixel gets its "typical" clear value.
        # Median (not mean) so a missed cloud or shadow can't drag the value.
        bands.append(refl.median("time", skipna=True))

    comp = xr.concat(bands, dim="band").assign_coords(band=["blue", "green", "red", "nir"])
    clear_count = good.sum("time")              # how many clear looks each pixel got

    # Store as uint16 (same units as Sentinel-2, 0 = no data) to keep files small.
    comp = comp.fillna(0).round().astype("uint16").rio.write_crs(CRS).rio.write_nodata(0)
    out = DATA_DIR / f"composite_{name}.tif"
    with dask.config.set(scheduler="threads", num_workers=4):
        comp.rio.to_raster(out, tiled=True, compress="deflate", lock=threading.Lock())
        n_clear = clear_count.compute().values
    print(f"[{name}] saved {out}  ({comp.sizes['y']} x {comp.sizes['x']} px)")
    print(f"[{name}] pixels with 0 clear looks: {(n_clear == 0).mean():.2%}   "
          f"median clear looks per pixel: {np.median(n_clear):.0f}")


if __name__ == "__main__":
    import sys

    import rioxarray  # noqa: F401  (adds the .rio accessor to xarray)
    # Optional: pass "before" or "after" to rebuild just one period.
    for name in sys.argv[1:] or DATES:
        build_composite(name)

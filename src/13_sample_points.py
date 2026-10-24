"""Phase 4a: pick random points for a proper accuracy check, and build a labelling page.

Why: comparing with Hansen only told us how much we agree with *another map*. The
standard way to measure a map's accuracy (Olofsson et al. 2014) is to look at random
points yourself and record what really happened there.

Stratified random sampling: the map's classes have very different sizes (cleared is
~3% of the area). A plain random sample of 300 points would contain only ~9 cleared
points, far too few. So we sample each map class ("stratum") separately and later
weight every stratum by its true share of the area, which keeps the estimates unbiased.

The labeller is BLIND: the page never shows what the model said, so it can't sway you.

Run:  .venv/Scripts/python src/13_sample_points.py
Makes: validation/samples.csv (point id, stratum, location; the answer key, don't open it
       while labelling), validation/labeler.html + validation/chips/*.jpg
"""
import csv
import json

import numpy as np
import rioxarray
from PIL import Image
from pyproj import Transformer

from config import CRS, DATA_DIR, ROOT

SEED = 2026
PER_STRATUM = {"loss": 100, "stable_forest": 80, "stable_nonforest": 70, "gain": 50}
CLOSE, WIDE = 30, 150         # half-widths in pixels: 600 m close-up, 3 km context
OUT = ROOT / "validation"
(OUT / "chips").mkdir(parents=True, exist_ok=True)

change = rioxarray.open_rasterio(DATA_DIR / "change_unet.tif").values[0]
pb = rioxarray.open_rasterio(DATA_DIR / "forest_prob_before.tif").values[0]
pa = rioxarray.open_rasterio(DATA_DIR / "forest_prob_after.tif").values[0]

# Map class of every pixel. The four strata cover the whole valid area exactly once.
valid = change != 255
strata = {
    "loss": change == 1,
    "gain": change == 2,
    "stable_forest": valid & (change == 0) & (pb > 50) & (pa > 50),
}
strata["stable_nonforest"] = valid & (change == 0) & ~strata["stable_forest"]

# Keep points far enough from the border that the wide chip fits.
H, W = change.shape
inner = np.zeros_like(valid)
inner[WIDE:H - WIDE, WIDE:W - WIDE] = True

rng = np.random.default_rng(SEED)
points = []
for name, n in PER_STRATUM.items():
    rows, cols = np.nonzero(strata[name] & inner)
    pick = rng.choice(len(rows), n, replace=False)
    points += [(name, int(rows[i]), int(cols[i])) for i in pick]
order = rng.permutation(len(points))          # mix strata so nothing gives them away
points = [points[i] for i in order]

stratum_pixels = {name: int(m.sum()) for name, m in strata.items()}
print("stratum sizes (ha):", {k: round(v * 0.01) for k, v in stratum_pixels.items()})


def natural_color(name):
    x = rioxarray.open_rasterio(DATA_DIR / f"{name}.tif").values[:3]
    return x, rioxarray.open_rasterio(DATA_DIR / f"{name}.tif").rio.transform()


def chip(img, r, c, half, size):
    x = img[:, r - half:r + half + 1, c - half:c + half + 1].astype("float32")
    rgb = np.clip(np.dstack([x[2], x[1], x[0]]) / 1800, 0, 1) ** (1 / 1.6)
    im = Image.fromarray((rgb * 255).astype("uint8"))
    return im.resize((size, size), Image.NEAREST)


to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
rows_out = []
for year, name in [("2021", "composite_before"), ("2026", "composite_after_norm")]:
    img, transform = natural_color(name)
    for k, (stratum, r, c) in enumerate(points):
        chip(img, r, c, CLOSE, 305).save(OUT / "chips" / f"{k:03d}_{year}_close.jpg", quality=88)
        chip(img, r, c, WIDE, 301).save(OUT / "chips" / f"{k:03d}_{year}_wide.jpg", quality=85)
        if year == "2021":
            x, y = transform * (c + 0.5, r + 0.5)
            lon, lat = to_lonlat.transform(x, y)
            rows_out.append({"id": k, "stratum": stratum, "row": r, "col": c,
                             "lat": round(lat, 6), "lon": round(lon, 6)})
    del img

with open(OUT / "samples.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
json.dump(stratum_pixels, open(OUT / "strata_pixels.json", "w"), indent=1)

# The page only gets id + location: no stratum, no model output.
page_points = [{"id": p["id"], "lat": p["lat"], "lon": p["lon"]} for p in rows_out]
html = (ROOT / "src" / "labeler_template.html").read_text(encoding="utf-8")
(OUT / "labeler.html").write_text(html.replace("__POINTS__", json.dumps(page_points)), encoding="utf-8")
print(f"Saved {len(points)} points -> {OUT / 'labeler.html'}")

"""Phase 6b: a FRESH set of 300 points to compare v1 (WorldCover labels) and v3 (MapBiomas).

The first 300 points were already used to judge v1 and v2, so picking a winner on them
again would start tuning to the test. New points keep the comparison honest.

Strata are built from BOTH maps, so the places that decide the comparison get enough
points:
    both_loss       both maps say cleared
    v1_only_loss    only v1 says cleared   (v3 thinks this was a false alarm)
    v3_only_loss    only v3 says cleared
    edge            within 100 m of a forest edge in either 2021 map, not cleared in
                    either: where missed clearings hide (this narrows the recall range)
    forest          forest interior in either 2021 map, not cleared in either
    other           everything else
Same blind labelling page as round 1 (it never shows what either model said).

Run:  .venv/Scripts/python src/13b_sample_round2.py
Makes: validation2/samples.csv, validation2/strata_pixels.json, validation2/labeler.html + chips
"""
import csv
import json

import numpy as np
import rioxarray
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

from config import CRS, DATA_DIR, ROOT

SEED = 777
PER_STRATUM = {"both_loss": 50, "v1_only_loss": 60, "v3_only_loss": 60, "edge": 60, "forest": 35, "other": 35}
CLOSE, WIDE = 30, 150
EDGE_PX = 10                       # 100 m
OUT = ROOT / "validation2"
(OUT / "chips").mkdir(parents=True, exist_ok=True)


def read(name):
    return rioxarray.open_rasterio(DATA_DIR / f"{name}.tif").values[0]


c1, c3 = read("change_unet"), read("change_unet_v3")
f1, f3 = read("forest_prob_before") > 50, read("forest_prob_before_v3") > 50
valid = (c1 != 255) & (c3 != 255)
l1, l3 = (c1 == 1) & valid, (c3 == 1) & valid
forest_any = (f1 | f3) & valid
# Edge zone: within 100 m of a forest / non-forest boundary in either 2021 map.
edge_zone = np.zeros_like(valid)
for f in (f1, f3):
    grown = ndimage.binary_dilation(f, iterations=EDGE_PX)
    shrunk = ndimage.binary_erosion(f, iterations=EDGE_PX)
    edge_zone |= grown & ~shrunk
no_loss = valid & ~l1 & ~l3
strata = {
    "both_loss": l1 & l3,
    "v1_only_loss": l1 & ~l3,
    "v3_only_loss": l3 & ~l1,
    "edge": no_loss & edge_zone,
    "forest": no_loss & ~edge_zone & forest_any,
}
strata["other"] = no_loss & ~edge_zone & ~forest_any
assert sum(m.sum() for m in strata.values()) == valid.sum()
stratum_pixels = {k: int(m.sum()) for k, m in strata.items()}
print("stratum sizes (ha):", {k: round(v * 0.01) for k, v in stratum_pixels.items()})

H, W = valid.shape
inner = np.zeros_like(valid)
inner[WIDE:H - WIDE, WIDE:W - WIDE] = True
rng = np.random.default_rng(SEED)
points = []
for name, n in PER_STRATUM.items():
    rows, cols = np.nonzero(strata[name] & inner)
    pick = rng.choice(len(rows), n, replace=False)
    points += [(name, int(rows[i]), int(cols[i])) for i in pick]
points = [points[i] for i in rng.permutation(len(points))]


def chip(img, r, c, half, size):
    x = img[:, r - half:r + half + 1, c - half:c + half + 1].astype("float32")
    rgb = np.clip(np.dstack([x[2], x[1], x[0]]) / 1800, 0, 1) ** (1 / 1.6)
    return Image.fromarray((rgb * 255).astype("uint8")).resize((size, size), Image.NEAREST)


to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
rows_out = []
for year, name in [("2021", "composite_before"), ("2026", "composite_after_norm")]:
    da = rioxarray.open_rasterio(DATA_DIR / f"{name}.tif")
    img, transform = da.values[:3], da.rio.transform()
    for k, (stratum, r, c) in enumerate(points):
        chip(img, r, c, CLOSE, 305).save(OUT / "chips" / f"{k:03d}_{year}_close.jpg", quality=88)
        chip(img, r, c, WIDE, 301).save(OUT / "chips" / f"{k:03d}_{year}_wide.jpg", quality=85)
        if year == "2021":
            lon, lat = to_lonlat.transform(*(transform * (c + 0.5, r + 0.5)))
            rows_out.append({"id": k, "stratum": stratum, "row": r, "col": c,
                             "lat": round(lat, 6), "lon": round(lon, 6)})
    del img

with open(OUT / "samples.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
json.dump(stratum_pixels, open(OUT / "strata_pixels.json", "w"), indent=1)

page_points = [{"id": p["id"], "lat": p["lat"], "lon": p["lon"]} for p in rows_out]
html = (ROOT / "src" / "labeler_template.html").read_text(encoding="utf-8")
html = (html.replace("__POINTS__", json.dumps(page_points))
            .replace("forest-labels-v2", "forest-labels-round2")      # separate save slot
            .replace('a.download = "labels.csv"', 'a.download = "labels_round2.csv"')
            .replace("<title>Forest Point Labeler</title>", "<title>Forest Labeler Round 2</title>"))
(OUT / "labeler.html").write_text(html, encoding="utf-8")
print(f"Saved {len(points)} points -> {OUT / 'labeler.html'}")

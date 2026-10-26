"""Phase 4b: honest accuracy numbers from the hand-labelled points (Olofsson et al. 2014).

Every sampled point stands for a share of the map: a "loss" point stands for
8,705 ha / 100 points, a "stable non-forest" point for 180,590 ha / 70 points. Weighting
each point by the area it represents gives unbiased estimates of:
  - user's accuracy  (precision): of what the map calls cleared, how much really was?
  - producer's accuracy (recall): of all real clearing, how much did the map catch?
  - the real cleared AREA, with a 95% confidence interval. This can differ from the
    map's 8,705 ha: the map's misses add area, its false alarms remove area.
Confidence intervals come from a stratified bootstrap: resample points within each
stratum 5,000 times and see how much the numbers wobble.

The strata come from the v1 map (that's how the points were drawn), but the estimators
work for ANY map: for v2 we just read the v2 map's class at each point. The labels
describe the ground, not a map, so they don't need redoing.

Run:  .venv/Scripts/python src/14_accuracy.py [v2] [path/to/labels.csv]
"""
import csv
import json
import sys

import numpy as np
import rioxarray

from config import DATA_DIR, ROOT

PIXEL_HA = 0.01
CLASSES = ["cleared", "regrew", "stayed_forest", "never_forest"]

args = sys.argv[1:]
V = "_v2" if "v2" in args else ""
csv_args = [a for a in args if a.endswith(".csv")]
labels_path = csv_args[0] if csv_args else ROOT / "validation" / "labels.csv"

samples = {int(r["id"]): r for r in csv.DictReader(open(ROOT / "validation" / "samples.csv"))}
labels = {int(r["id"]): r["label"] for r in csv.DictReader(open(labels_path))}
area = {k: v * PIXEL_HA for k, v in json.load(open(ROOT / "validation" / "strata_pixels.json")).items()}
total_area = sum(area.values())

# What this version's map says at each point.
change = rioxarray.open_rasterio(DATA_DIR / f"change_unet{V}.tif").values[0]
pb = rioxarray.open_rasterio(DATA_DIR / f"forest_prob_before{V}.tif").values[0]
pa = rioxarray.open_rasterio(DATA_DIR / f"forest_prob_after{V}.tif").values[0]


def map_class(r, c):
    if change[r, c] == 1:
        return "cleared"
    if change[r, c] == 2:
        return "regrew"
    return "stayed_forest" if pb[r, c] > 50 and pa[r, c] > 50 else "never_forest"


used = [i for i, l in labels.items() if l != "cant_tell"]
print(f"map: change_unet{V}.tif   {len(labels)} points labelled, "
      f"{len(labels) - len(used)} 'can't tell' left out, {len(samples) - len(labels)} not labelled yet\n")

# Per stratum: (reference label, map label) for every usable point.
pairs = {s: [] for s in area}
for i in used:
    s = samples[i]
    pairs[s["stratum"]].append((labels[i], map_class(int(s["row"]), int(s["col"]))))

print("What you saw (rows) vs what the map said (columns), point counts:")
print(f"{'':<16}" + "".join(f"{c:>15}" for c in CLASSES))
for ref in CLASSES:
    row = [sum(1 for g in pairs.values() for r, m in g if r == ref and m == c) for c in CLASSES]
    print(f"{ref:<16}" + "".join(f"{n:>15}" for n in row))
print()


def estimates(groups):
    """groups: stratum -> (ref, map) array. Returns precision, recall, true cleared ha, overall acc."""
    W = {s: area[s] / total_area for s in groups}
    both = sum(W[s] * np.mean((g[:, 0] == "cleared") & (g[:, 1] == "cleared")) for s, g in groups.items())
    mapped = sum(W[s] * np.mean(g[:, 1] == "cleared") for s, g in groups.items())
    truth = sum(W[s] * np.mean(g[:, 0] == "cleared") for s, g in groups.items())
    agree = sum(W[s] * np.mean(g[:, 0] == g[:, 1]) for s, g in groups.items())
    return both / mapped, both / truth, truth * total_area, agree


groups = {s: np.array(g) for s, g in pairs.items()}
point = estimates(groups)
rng = np.random.default_rng(0)
boot = np.array([estimates({s: g[rng.integers(0, len(g), len(g))] for s, g in groups.items()})
                 for _ in range(5000)])
lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

mapped_ha = (change == 1).sum() * PIXEL_HA
names = ["Precision (user's accuracy, cleared)", "Recall (producer's accuracy, cleared)",
         "Real cleared area (ha)", "Overall accuracy (4 classes, area weighted)"]
for k, name in enumerate(names):
    if k == 2:
        print(f"{name:<46} {point[k]:>8,.0f}   95% CI {lo[k]:,.0f} to {hi[k]:,.0f}"
              f"   (this map says {mapped_ha:,.0f})")
    else:
        print(f"{name:<46} {point[k]:>8.1%}   95% CI {lo[k]:.1%} to {hi[k]:.1%}")

json.dump({"map": f"change_unet{V}.tif", "mapped_cleared_ha": float(mapped_ha),
           "n_labelled": len(labels), "n_used": len(used),
           **{key: {"estimate": float(point[k]), "ci95": [float(lo[k]), float(hi[k])]}
              for k, key in enumerate(["precision", "recall", "cleared_ha", "overall_accuracy"])}},
          open(ROOT / "validation" / f"accuracy{V}.json", "w"), indent=1)
print(f"\nSaved {ROOT / 'validation' / f'accuracy{V}.json'}")

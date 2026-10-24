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

Run:  .venv/Scripts/python src/14_accuracy.py [path/to/labels.csv]
"""
import csv
import json
import sys

import numpy as np

from config import ROOT

PIXEL_HA = 0.01
MAP_SAYS = {"loss": "cleared", "gain": "regrew", "stable_forest": "stayed_forest",
            "stable_nonforest": "never_forest"}
REF = ["cleared", "stayed_forest", "never_forest", "regrew"]

labels_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "validation" / "labels.csv"
samples = {int(r["id"]): r for r in csv.DictReader(open(ROOT / "validation" / "samples.csv"))}
labels = {int(r["id"]): r["label"] for r in csv.DictReader(open(labels_path))}
area = {k: v * PIXEL_HA for k, v in json.load(open(ROOT / "validation" / "strata_pixels.json")).items()}
total_area = sum(area.values())

used = {i: l for i, l in labels.items() if l != "cant_tell"}
print(f"{len(labels)} points labelled, {len(labels) - len(used)} 'can't tell' left out, "
      f"{len(samples) - len(labels)} not labelled yet\n")

# Per stratum: which points, and what the reference says.
by_stratum = {s: [] for s in MAP_SAYS}
for i, l in used.items():
    by_stratum[samples[i]["stratum"]].append(l)

print("What you saw (rows) vs what the map said (columns), point counts:")
print(f"{'':<16}" + "".join(f"{MAP_SAYS[s]:>15}" for s in MAP_SAYS))
for ref in REF:
    print(f"{ref:<16}" + "".join(f"{by_stratum[s].count(ref):>15}" for s in MAP_SAYS))
print()


def estimates(groups):
    """groups: stratum -> array of reference labels. Returns UA, PA, cleared area, overall acc."""
    cleared_share = {s: np.mean(g == "cleared") for s, g in groups.items()}
    agree = {s: np.mean(g == MAP_SAYS[s]) for s, g in groups.items()}
    true_cleared_ha = sum(area[s] * cleared_share[s] for s in groups)
    ua = cleared_share["loss"]
    pa = area["loss"] * cleared_share["loss"] / true_cleared_ha
    oa = sum(area[s] * agree[s] for s in groups) / total_area
    return ua, pa, true_cleared_ha, oa


groups = {s: np.array(g) for s, g in by_stratum.items()}
point = estimates(groups)
rng = np.random.default_rng(0)
boot = np.array([estimates({s: rng.choice(g, len(g)) for s, g in groups.items()})
                 for _ in range(5000)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

names = ["Precision (user's accuracy, cleared)", "Recall (producer's accuracy, cleared)",
         "Real cleared area (ha)", "Overall accuracy (4 classes, area weighted)"]
for k, name in enumerate(names):
    if k == 2:
        print(f"{name:<46} {point[k]:>8,.0f}   95% CI {lo[k]:,.0f} to {hi[k]:,.0f}"
              f"   (map says {area['loss']:,.0f})")
    else:
        print(f"{name:<46} {point[k]:>8.1%}   95% CI {lo[k]:.1%} to {hi[k]:.1%}")

json.dump({"n_labelled": len(labels), "n_used": len(used),
           "counts": {s: {r: by_stratum[s].count(r) for r in REF} for s in MAP_SAYS},
           **{key: {"estimate": float(point[k]), "ci95": [float(lo[k]), float(hi[k])]}
              for k, key in enumerate(["precision", "recall", "cleared_ha", "overall_accuracy"])}},
          open(ROOT / "validation" / "accuracy.json", "w"), indent=1)
print(f"\nSaved {ROOT / 'validation' / 'accuracy.json'}")

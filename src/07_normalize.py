"""Phase 3a: make 2026 brightness comparable to 2021 (relative radiometric normalization).

Idea: some pixels should look the same in both years (old-growth forest that was never
cut, town roofs and roads). If they don't, the difference is haze/sensor/processing,
not real change. We fit, per band, a straight line  value_2021 = a * value_2026 + b
on those "pseudo-invariant" pixels, then apply it to the whole 2026 image.

Run:  .venv/Scripts/python src/07_normalize.py
Makes: data/composite_after_norm.tif, outputs/07_normalization.png
"""
import matplotlib.pyplot as plt
import numpy as np
import rioxarray

from config import DATA_DIR, OUT_DIR

BANDS = ["blue", "green", "red", "nir"]
STEP = 4  # pick reference pixels from every 4th row/column (plenty, and light on RAM)

before = rioxarray.open_rasterio(DATA_DIR / "composite_before.tif")
after = rioxarray.open_rasterio(DATA_DIR / "composite_after.tif")
labels = rioxarray.open_rasterio(DATA_DIR / "labels_2021.tif").values[0, ::STEP, ::STEP]

b = before.values[:, ::STEP, ::STEP].astype("float32")
a = after.values[:, ::STEP, ::STEP].astype("float32")


def ndvi(x):
    return (x[3] - x[2]) / (x[3] + x[2] + 1e-6)


# Invariant pixels: dense forest in BOTH years (dark in visible light), plus built-up
# areas (bright). Having both dark and bright anchors makes the fitted line reliable
# across the whole brightness range, not just for forest-like values.
stable_forest = (labels == 1) & (ndvi(b) > 0.8) & (ndvi(a) > 0.8)
# labels_2021.tif only says forest yes/no, so find built-up/bare ground by a rule:
# barely any vegetation in both years and bright in visible red.
built = (ndvi(b) < 0.2) & (ndvi(a) < 0.2) & (b[2] > 1200) & (a[2] > 1200)
ref = (stable_forest | built) & (b[3] > 0) & (a[3] > 0)
print(f"reference pixels: {stable_forest.sum():,} stable forest + {built.sum():,} bright built-up/bare")

coeffs = {}
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for i, (band, ax) in enumerate(zip(BANDS, axes)):
    x, y = a[i][ref], b[i][ref]
    # Reduced major axis regression: both years are noisy, so don't use ordinary least
    # squares (it assumes x is exact and flattens the slope when x's range is narrow,
    # which happened for NIR: slope 0.52). RMA slope = ratio of spreads.
    slope = np.sign(np.corrcoef(x, y)[0, 1]) * y.std() / x.std()
    intercept = y.mean() - slope * x.mean()
    coeffs[band] = (slope, intercept)
    print(f"{band:<5}: 2021 = {slope:.3f} x 2026 {intercept:+.1f}    "
          f"(median before/after on reference: {np.median(y):.0f} / {np.median(x):.0f})")
    pick = np.random.default_rng(0).choice(len(x), min(5000, len(x)), replace=False)
    ax.scatter(x[pick], y[pick], s=2, alpha=0.3)
    lim = [0, np.percentile(np.r_[x, y], 99.5)]
    ax.plot(lim, lim, "k--", lw=1, label="no change (y = x)")
    ax.plot(lim, [slope * lim[0] + intercept, slope * lim[1] + intercept], "r", lw=2, label="fitted line")
    ax.set(title=band, xlabel="2026 value", ylabel="2021 value", xlim=lim, ylim=lim)
    ax.legend(fontsize=8)
plt.suptitle("Reference pixels: if 2021 and 2026 matched, dots would sit on the dashed line", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_normalization.png", dpi=90)

# Apply to the full-resolution 2026 image, band by band to keep memory low.
out = after.copy()
valid = (after.values > 0).all(axis=0)
for i, band in enumerate(BANDS):
    slope, intercept = coeffs[band]
    fixed = np.clip(after.values[i].astype("float32") * slope + intercept, 1, 65535)
    out.values[i] = np.where(valid, fixed, 0).round().astype("uint16")
out.rio.to_raster(DATA_DIR / "composite_after_norm.tif", tiled=True, compress="deflate")

n = out.values[:, ::STEP, ::STEP].astype("float32")
print(f"\nNDVI > 0.61 share   2021: {(ndvi(b) > 0.61).mean():.1%}   "
      f"2026 raw: {(ndvi(a) > 0.61).mean():.1%}   2026 normalized: {(ndvi(n) > 0.61).mean():.1%}")
print(f"Saved {DATA_DIR / 'composite_after_norm.tif'} and {OUT_DIR / '07_normalization.png'}")

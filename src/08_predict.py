"""Phase 3b: run the U-Net over the full 55 x 55 km image for each year.

The model was trained on 256x256 tiles, so we slide a 256x256 window across the big
image. Windows overlap (step 192) and their predictions are blended with weights that
fade towards each window's edge: a pixel near a window's border has little context
around it, so its prediction is less trustworthy than one near the centre.

Run:  .venv/Scripts/python src/08_predict.py         (v1: 4 bands)
      .venv/Scripts/python src/08_predict.py v2      (v2: 6 bands with SWIR)
Makes: data/forest_prob_{before,after,after_raw}.tif   (0-100 = % forest probability, 255 = no data)
       data/forest_prob_{before,after}_v2.tif
"""
import sys
import time

import numpy as np
import rioxarray
import torch

from config import DATA_DIR
from model import load_model, to_features

TILE, STEP, BATCH = 256, 192, 8

VERSION = "v2" if "v2" in sys.argv[1:] else "v1"
model, mean, std = load_model(VERSION)

# Window weight: 2D bump, high in the middle, low (not zero) at the edges.
w1 = 0.1 + 0.9 * np.hanning(TILE)
weight = np.outer(w1, w1).astype("float32")


def predict(src_names, out_name):
    t0 = time.time()
    da = rioxarray.open_rasterio(DATA_DIR / f"{src_names[0]}.tif")
    img = np.concatenate([da.values] + [rioxarray.open_rasterio(DATA_DIR / f"{n}.tif").values
                                        for n in src_names[1:]])   # (4 or 6, H, W) uint16
    _, H, W = img.shape
    # Pad bottom/right (mirror) so the windows tile the image exactly.
    pad_h = (-(H - TILE)) % STEP
    pad_w = (-(W - TILE)) % STEP
    img_p = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = img_p.shape[1:]

    prob_sum = np.zeros((Hp, Wp), "float32")
    w_sum = np.zeros((Hp, Wp), "float32")
    corners = [(r, c) for r in range(0, Hp - TILE + 1, STEP) for c in range(0, Wp - TILE + 1, STEP)]
    with torch.no_grad():
        for k in range(0, len(corners), BATCH):
            batch = corners[k:k + BATCH]
            X = np.stack([img_p[:, r:r + TILE, c:c + TILE] for r, c in batch])
            p = torch.sigmoid(model(torch.from_numpy(to_features(X, mean, std)))).numpy()
            for (r, c), pi in zip(batch, p):
                prob_sum[r:r + TILE, c:c + TILE] += pi * weight
                w_sum[r:r + TILE, c:c + TILE] += weight
            if k % (BATCH * 40) == 0:
                print(f"   [{out_name}] {k}/{len(corners)} windows")

    prob = (prob_sum / w_sum)[:H, :W]
    out = np.round(prob * 100).astype("uint8")
    out[(img == 0).all(axis=0)] = 255
    da.isel(band=0).copy(data=out).rio.write_nodata(255).rio.to_raster(
        DATA_DIR / f"forest_prob_{out_name}.tif", compress="deflate")
    valid = out != 255
    print(f"[{out_name}] forest (prob > 50%): {(out[valid] > 50).mean():.1%} of area   "
          f"({len(corners)} windows, {time.time() - t0:.0f}s)")


if VERSION == "v1":
    predict(["composite_before"], "before")
    predict(["composite_after_norm"], "after")
    predict(["composite_after"], "after_raw")   # without normalization, to measure its effect
else:
    predict(["composite_before", "composite_before_swir"], "before_v2")
    predict(["composite_after_norm", "composite_after_swir_norm"], "after_v2")

"""Phase 3e: interactive web map of the results (a single self-contained HTML file).

Layers you can toggle: 2021 image, 2026 image, cleared patches (click for details),
regrowth, Hansen loss 2022-2025 for comparison.

Run:  .venv/Scripts/python src/11_map.py
Makes: outputs/deforestation_map.html   (open in any browser, no server needed)
"""
import base64
import io
import json

import folium
import numpy as np
import rioxarray
from PIL import Image
from rasterio.enums import Resampling

from config import DATA_DIR, OUT_DIR

STEP = 3            # display every 3rd pixel (30 m): sharp enough, keeps the file small


def to_latlon(path, resampling):
    """Reproject a raster to lat/lon (what web maps use) at display resolution."""
    da = rioxarray.open_rasterio(path)
    da = da.rio.reproject("EPSG:4326", resolution=0.00027, resampling=resampling)
    bounds = da.rio.bounds()  # (west, south, east, north)
    return da.values, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def data_url(rgba, fmt):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format=fmt, quality=85)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def natural_color_url(path):
    # Natural color (red, green, blue bands): forest dark green, pasture tan. Not false
    # color: there forest is red, and our red "cleared" patches would vanish on top of it.
    x, bounds = to_latlon(path, Resampling.average)
    x = x.astype("float32")
    rgb = np.clip(np.dstack([x[2], x[1], x[0]]) / 1800, 0, 1) ** (1 / 1.6)  # gamma brightens dark forest
    return data_url((rgb * 255).astype("uint8"), "JPEG"), bounds


def mask_url(mask, color):
    rgba = np.zeros(mask.shape + (4,), "uint8")
    rgba[mask] = color
    return data_url(rgba, "PNG")


img21, bounds = natural_color_url(DATA_DIR / "composite_before.tif")
img26, _ = natural_color_url(DATA_DIR / "composite_after_norm.tif")
change, _ = to_latlon(DATA_DIR / "change_unet.tif", Resampling.nearest)
hansen = rioxarray.open_rasterio(DATA_DIR / "hansen_lossyear.tif").values[0]
hansen_bounds = rioxarray.open_rasterio(DATA_DIR / "hansen_lossyear.tif").rio.bounds()

patches = json.load(open(OUT_DIR / "loss_patches.geojson"))
total_ha = sum(f["properties"]["area_ha"] for f in patches["features"])
center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]

m = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite basemap (Esri)").add_to(m)
# Esri's street map, not OpenStreetMap: OSM blocks tiles for pages opened as local files.
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Street Map", name="Street map (Esri)", show=False).add_to(m)

folium.raster_layers.ImageOverlay(img21, bounds, name="Sentinel-2 Jul-Aug 2021",
                                  show=False).add_to(m)
folium.raster_layers.ImageOverlay(img26, bounds, name="Sentinel-2 Jul-Aug 2026",
                                  show=True).add_to(m)
folium.raster_layers.ImageOverlay(mask_url(change[0] == 2, (40, 110, 230, 200)), bounds,
                                  name="Regrowth 2021-2026 (blue)", show=False).add_to(m)
folium.raster_layers.ImageOverlay(
    mask_url((hansen >= 22) & (hansen <= 25), (255, 200, 0, 190)),
    [[hansen_bounds[1], hansen_bounds[0]], [hansen_bounds[3], hansen_bounds[2]]],
    name="Hansen tree cover loss 2022-2025 (yellow)", show=False).add_to(m)

folium.GeoJson(
    patches, name=f"Cleared 2021-2026 (red): {len(patches['features'])} patches",
    style_function=lambda f: {"color": "#ff1a1a", "weight": 1.5, "fillColor": "#ff1a1a", "fillOpacity": 0.45},
    highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.7},
    tooltip=folium.GeoJsonTooltip(["area_ha"], aliases=["Cleared (ha):"]),
    popup=folium.GeoJsonPopup(["patch_id", "area_ha", "forest_prob_2021", "forest_prob_2026"],
                              aliases=["Patch", "Area (ha)", "Forest prob. 2021 (%)", "Forest prob. 2026 (%)"]),
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element(f"""
<div style="position:fixed; bottom:24px; left:12px; z-index:9999; max-width:330px;
            background:rgba(255,255,255,0.93); padding:12px 14px; border-radius:8px;
            font:13px/1.45 system-ui, sans-serif; box-shadow:0 1px 6px rgba(0,0,0,0.3)">
  <b style="font-size:15px">Deforestation in Rondônia, Brazil, 2021 → 2026</b><br>
  <span style="color:#d00"><b>{total_ha:,.0f} ha</b></span> of forest cleared in
  {len(patches['features']):,} patches (≥ 0.5 ha), about 7% of the 2021 forest.<br>
  Sentinel-2 Jul–Aug composites · U-Net forest segmentation (test IoU 0.91)<br>
  <span style="color:#666">Click a red patch for details. Toggle layers top right.</span>
</div>"""))

out = OUT_DIR / "deforestation_map.html"
m.save(out)
print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")

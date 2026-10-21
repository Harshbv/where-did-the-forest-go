"""Project-wide settings. Every script imports from here so we change things in one place."""
from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Area of interest: a deforestation frontier in Rondonia, Brazil.
# Format is (min_lon, min_lat, max_lon, max_lat) in degrees (EPSG:4326).
BBOX = (-63.5, -10.0, -63.0, -9.5)

# Rondonia sits in UTM zone 20 South. Working in meters (not degrees)
# makes every pixel exactly 10 m x 10 m, so areas in hectares are easy.
CRS = "EPSG:32720"

# Peak dry season = fewest clouds over the Amazon.
# "before" must be 2021: that's the year of our WorldCover training labels.
# "after" is the latest complete dry season.
# Both use the SAME months: pasture is green in June and brown by August, so
# comparing a June image with an August one would "detect" seasons, not clearing.
DATES = {
    "before": "2021-07-01/2021-08-31",
    "after": "2026-07-01/2026-08-31",
}

# Planetary Computer STAC catalog
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Bands we use. Asset names are how Planetary Computer labels each band file.
BANDS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",   # near-infrared: forests are very bright here
    "SCL": "scl",   # scene classification layer: tells us which pixels are cloud
}

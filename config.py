"""
config.py — Central configuration for Climate AI v2 (Production Grid Edition)

All region, model, and API settings are defined here.
Grid-based: replaces static city list with a dynamic coordinate matrix.
"""
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─── Google Earth Engine ────────────────────────────────────────────────────
GEE_SERVICE_ACCOUNT   = os.getenv("GEE_SERVICE_ACCOUNT", "")
GEE_KEY_FILE          = os.getenv("GEE_KEY_FILE", "gee-key.json")

# ─── Grid Configuration ─────────────────────────────────────────────────────
# 5x5 grid covering Central India (Madhya Pradesh + surrounding regions)
# Covers ~20°N–25°N, 75°E–80°E — highly diverse climate zones
_GRID_LATS = np.linspace(20.0, 28.0, 5)   # 5 latitude bands
_GRID_LONS = np.linspace(72.0, 88.0, 5)   # 5 longitude bands

GRID_POINTS: dict = {}
for _lat in _GRID_LATS:
    for _lon in _GRID_LONS:
        _key = f"Grid_{_lat:.1f}_{_lon:.1f}"
        _bbox = [round(_lon - 0.025, 4), round(_lat - 0.025, 4),
                 round(_lon + 0.025, 4), round(_lat + 0.025, 4)]
        GRID_POINTS[_key] = {"lat": float(_lat), "lon": float(_lon), "bbox": _bbox}

# Pre-computed numpy arrays for fast nearest-point lookup via scipy.cKDTree
GRID_COORDS = np.array([[v["lat"], v["lon"]] for v in GRID_POINTS.values()])
GRID_IDS    = list(GRID_POINTS.keys())

# Legacy compatibility: keep an alias so older references don't crash
FOCAL_CITIES = GRID_POINTS   # alias

# Entire India region bounding box (min_lon, min_lat, max_lon, max_lat)
INDIA_BBOX = [68.1, 8.0, 97.4, 37.6]

# ─── Data Collection Settings ───────────────────────────────────────────────
HISTORICAL_START  = "2018-01-01"
HISTORICAL_END    = "2024-12-31"
COLLECTION_SCALE  = 5000           # 5 km spatial resolution

GEE_DATASETS = {
    "ERA5_DAILY": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "MODIS_TEMP": "MODIS/061/MOD11A1",
    "GPM_RAIN":   "NASA/GPM_L3/IMERG_V06",
}

# ─── ERA5 Bands: 8 Features (upgraded from 6) ────────────────────────────────
ERA5_BANDS = {
    "temperature_2m_max":                    "temp_max",
    "temperature_2m_min":                    "temp_min",
    "total_precipitation_sum":               "precipitation",
    "u_component_of_wind_10m":               "wind_u",
    "v_component_of_wind_10m":               "wind_v",
    "surface_solar_radiation_downwards_sum": "solar_radiation",
    "volumetric_soil_water_layer_1":         "soil_moisture",   # NEW
    "leaf_area_index_high_vegetation":       "lai_high_veg",    # NEW
}

FEATURE_COLS = [
    "temp_max", "temp_min", "temp_mean",
    "precipitation", "wind_speed", "solar_radiation",
    "soil_moisture", "lai_high_veg",
]
N_FEATURES = len(FEATURE_COLS)

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/climate.duckdb")

# ─── ML Model Settings ───────────────────────────────────────────────────────
SEQ_LENGTH        = 30
FORECAST_HORIZON  = 10
ANOMALY_THRESHOLD = 0.65   # Base percentile — overridden per grid cell adaptively
BATCH_SIZE        = 32
EPOCHS            = 50
ENSEMBLE_SIZE     = 3      # Number of models in the forecasting ensemble

MODEL_SAVE_DIR    = "models/saved"
MLFLOW_TRACKING   = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

# SHAP sample sizes
SHAP_BG_SAMPLES   = 50    # Background reference samples for KernelExplainer
SHAP_EVAL_SAMPLES = 5     # Anomalous samples to evaluate per run

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = "logs/climate_ai.log"

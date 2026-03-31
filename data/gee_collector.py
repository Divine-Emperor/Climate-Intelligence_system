"""
data/gee_collector.py
─────────────────────
Google Earth Engine data collector — Grid Edition.

Fetches 8-band ERA5 daily climate variables at grid-point resolution.
Falls back gracefully to realistic mock data when GEE auth is unavailable.
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GEE_SERVICE_ACCOUNT, GEE_KEY_FILE, GEE_DATASETS,
    ERA5_BANDS, COLLECTION_SCALE, GRID_POINTS, INDIA_BBOX,
    HISTORICAL_START, HISTORICAL_END,
)

_GEE_AVAILABLE: Optional[bool] = None


def _init_gee() -> bool:
    global _GEE_AVAILABLE
    if _GEE_AVAILABLE is not None:
        return _GEE_AVAILABLE
    try:
        import ee
        if GEE_KEY_FILE and os.path.exists(GEE_KEY_FILE):
            credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_KEY_FILE)
            ee.Initialize(credentials)
            logger.info("GEE initialized via service account key.")
        else:
            ee.Initialize()
            logger.info("GEE initialized via user credentials.")
        _GEE_AVAILABLE = True
    except Exception as e:
        logger.warning(f"GEE init failed — using mock data: {e}")
        _GEE_AVAILABLE = False
    return _GEE_AVAILABLE


def fetch_era5_grid_point(
    grid_id: str,
    start_date: str = HISTORICAL_START,
    end_date: str = HISTORICAL_END,
) -> pd.DataFrame:
    """
    Fetch 8-band ERA5 Land daily climate data for a single grid point.
    Returns DataFrame with columns:
        [date, grid_id, lat, lon, temp_max, temp_min, temp_mean,
         precipitation, wind_speed, solar_radiation, soil_moisture, lai_high_veg]
    """
    gee_available = _init_gee()
    grid_info = GRID_POINTS.get(grid_id)
    if not grid_info:
        raise ValueError(f"Grid point '{grid_id}' not found in GRID_POINTS config.")

    if gee_available:
        return _fetch_from_gee(grid_id, grid_info, start_date, end_date)
    else:
        logger.warning(f"GEE unavailable — generating mock data for {grid_id}.")
        return _generate_mock_data(grid_id, grid_info, start_date, end_date)


def fetch_all_grid_points(
    start_date: str = HISTORICAL_START,
    end_date: str = HISTORICAL_END,
) -> pd.DataFrame:
    """Fetch 8-band ERA5 data for ALL configured grid points and concatenate."""
    frames = []
    grid_ids = list(GRID_POINTS.keys())
    logger.info(f"Fetching data for {len(grid_ids)} grid points ...")
    for grid_id in grid_ids:
        try:
            df = fetch_era5_grid_point(grid_id, start_date, end_date)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {grid_id}: {e}")
    if not frames:
        raise RuntimeError("Could not collect data for any grid point.")
    return pd.concat(frames, ignore_index=True)


# ─── Legacy alias ─────────────────────────────────────────────────────────────
def fetch_all_cities(start_date=HISTORICAL_START, end_date=HISTORICAL_END):
    """Backward-compatibility alias → fetch_all_grid_points."""
    return fetch_all_grid_points(start_date, end_date)


# ─── GEE private implementation ──────────────────────────────────────────────

def _fetch_from_gee(grid_id, grid_info, start_date, end_date) -> pd.DataFrame:
    import ee
    bbox = grid_info["bbox"]
    region = ee.Geometry.Rectangle(bbox)

    era5 = (
        ee.ImageCollection(GEE_DATASETS["ERA5_DAILY"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(list(ERA5_BANDS.keys()))
    )

    def extract_date(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=COLLECTION_SCALE,
            maxPixels=1e9,
        )
        return image.set(stats).set("date", image.date().format("YYYY-MM-dd"))

    collection = era5.map(extract_date)
    dates_list = collection.aggregate_array("date").getInfo()

    rows = []
    for idx, date_str in enumerate(dates_list):
        image = ee.Image(collection.toList(len(dates_list)).get(idx))
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=COLLECTION_SCALE,
            maxPixels=1e9,
        ).getInfo()
        row = {
            "date": date_str,
            "grid_id": grid_id,
            "lat": grid_info["lat"],
            "lon": grid_info["lon"],
        }
        for gee_band, col_name in ERA5_BANDS.items():
            val = stats.get(gee_band)
            if "temperature" in gee_band and val is not None:
                val = val - 273.15  # Kelvin → Celsius
            row[col_name] = round(val, 4) if val is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"]       = pd.to_datetime(df["date"])
    df["temp_mean"]  = (df["temp_max"] + df["temp_min"]) / 2
    df["wind_speed"] = np.sqrt(df.get("wind_u", 0) ** 2 + df.get("wind_v", 0) ** 2)
    df.drop(columns=["wind_u", "wind_v"], inplace=True, errors="ignore")
    return df


# ─── Mock data fallback ──────────────────────────────────────────────────────

def _generate_mock_data(grid_id, grid_info, start_date, end_date) -> pd.DataFrame:
    """Generate realistic 8-feature synthetic climate data for development."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n     = len(dates)
    seed  = int(abs(grid_info["lat"] * 1000 + grid_info["lon"] * 100)) % (2**31)
    np.random.seed(seed)

    lat = grid_info["lat"]
    # Temperature baseline based on latitude (cooler further north)
    base_temp = 32 - (lat - 20) * 0.8

    doy = dates.day_of_year
    temp_seasonal = base_temp + 10 * np.sin(2 * np.pi * (doy - 80) / 365)

    anomaly_mask = np.random.rand(n) < 0.06
    heatwave     = np.where(anomaly_mask, np.random.uniform(5, 14, n), 0)

    temp_max  = temp_seasonal + heatwave + np.random.normal(3, 1.5, n)
    temp_min  = temp_seasonal + heatwave - np.random.normal(6, 1.5, n)
    temp_mean = (temp_max + temp_min) / 2

    monsoon      = (doy >= 152) & (doy <= 274)
    precip_base  = np.where(monsoon, 8, 1)
    precipitation = np.maximum(0, np.random.exponential(precip_base, n))
    precipitation += np.where(anomaly_mask, np.random.uniform(20, 80, n), 0)

    wind_speed    = np.abs(np.random.normal(3.5, 1.2, n))
    solar_rad     = 15e6 * np.sin(np.pi * doy / 365) + np.random.normal(0, 5e5, n)

    # New features: soil moisture (higher during monsoon) and LAI (greenness)
    soil_moisture = np.clip(
        0.25 + np.where(monsoon, 0.2, 0) + np.random.normal(0, 0.05, n),
        0.05, 0.65
    )
    lai_high_veg  = np.clip(
        2.0 + np.where(monsoon, 2.0, 0) + np.random.normal(0, 0.3, n),
        0.1, 6.0
    )

    return pd.DataFrame({
        "date":            dates,
        "grid_id":         grid_id,
        "lat":             grid_info["lat"],
        "lon":             grid_info["lon"],
        "temp_max":        np.round(temp_max, 2),
        "temp_min":        np.round(temp_min, 2),
        "temp_mean":       np.round(temp_mean, 2),
        "precipitation":   np.round(np.clip(precipitation, 0, None), 2),
        "wind_speed":      np.round(wind_speed, 2),
        "solar_radiation": np.round(np.clip(solar_rad, 0, None), 0),
        "soil_moisture":   np.round(soil_moisture, 4),
        "lai_high_veg":    np.round(lai_high_veg, 3),
        "is_mock":         True,
    })

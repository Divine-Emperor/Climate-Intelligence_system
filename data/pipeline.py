"""
data/pipeline.py
────────────────
Data pipeline orchestrator — Grid Edition.

  1. Fetch 8-band ERA5 data from GEE for all grid points
  2. Validate and normalize
  3. Store in DuckDB (grid_id-keyed schema)
  4. Trigger anomaly scoring and forecasting
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID_POINTS, HISTORICAL_START, LOG_FILE, LOG_LEVEL, FEATURE_COLS
from data.gee_collector import fetch_all_grid_points, fetch_era5_grid_point
from data.data_store import upsert_climate_data, get_db_stats, get_climate_data

os.makedirs("logs", exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level=LOG_LEVEL)

VALID_RANGES = {
    "temp_max":        (-10, 60),
    "temp_min":        (-20, 45),
    "temp_mean":       (-15, 55),
    "precipitation":   (0, 500),
    "wind_speed":      (0, 50),
    "solar_radiation": (0, 4e7),
    "soil_moisture":   (0, 1),
    "lai_high_veg":    (0, 10),
}


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clamp all climate variables to sensible physical ranges."""
    original_len = len(df)
    id_col = "grid_id" if "grid_id" in df.columns else "city"
    df = df.dropna(subset=["date", id_col, "temp_mean"])

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                logger.warning(f"Clamping {out_of_range} out-of-range values in '{col}'.")
            df[col] = df[col].clip(lo, hi)

    dropped = original_len - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid rows during validation.")
    return df.reset_index(drop=True)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features needed by the ML models."""
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["day_of_year"]= df["date"].dt.day_of_year
    df["month"]      = df["date"].dt.month
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["heat_index"] = (df["temp_mean"]
                        + 0.33 * df.get("precipitation", 0)
                        - 0.7  * df.get("wind_speed", 0))
    return df


# ─── Pipeline steps ──────────────────────────────────────────────────────────

def run_initial_load():
    """Full historical load — run once on first setup."""
    logger.info("Starting FULL HISTORICAL grid load from GEE ...")
    df = fetch_all_grid_points(HISTORICAL_START)
    df = validate_dataframe(df)
    rows = upsert_climate_data(df)
    df   = enrich_features(df)
    logger.info(f"Initial grid load complete. Stored {rows} records.")
    return df


def run_daily_update():
    """Incremental daily update — fetch last 5 days for all grid points."""
    window_start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
    yesterday    = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Running DAILY UPDATE from {window_start} to {yesterday} ...")
    df   = fetch_all_grid_points(window_start, yesterday)
    df   = validate_dataframe(df)
    rows = upsert_climate_data(df)
    df   = enrich_features(df)
    logger.info(f"Daily update complete. Stored {rows} records.")
    return df


def run_grid_update(grid_id: str):
    """Incremental update for a single grid point."""
    existing = get_climate_data(grid_id)
    if len(existing) == 0:
        start = HISTORICAL_START
    else:
        latest = pd.to_datetime(existing["date"]).max()
        start  = (latest + timedelta(days=1)).strftime("%Y-%m-%d")

    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    if start > yesterday:
        logger.info(f"{grid_id} is already up-to-date.")
        return pd.DataFrame()

    logger.info(f"Updating {grid_id} from {start} to {yesterday} ...")
    df   = fetch_era5_grid_point(grid_id, start, yesterday)
    df   = validate_dataframe(df)
    rows = upsert_climate_data(df)
    df   = enrich_features(df)
    logger.info(f"{grid_id} update done. Stored {rows} records.")
    return df


def print_db_status():
    stats = get_db_stats()
    logger.info("─── Database Status ───────────────────────────────")
    logger.info(f"  Climate records  : {stats['climate_rows']:,}")
    logger.info(f"  Anomaly records  : {stats['anomaly_rows']:,}")
    logger.info(f"  Forecast records : {stats['forecast_rows']:,}")
    logger.info(f"  Grid points      : {len(stats.get('grid_ids', []))}")
    logger.info(f"  Date range       : {stats['date_range']}")
    logger.info("────────────────────────────────────────────────────")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Climate AI Grid Data Pipeline")
    parser.add_argument("--mode",    choices=["init", "update", "status"], default="update")
    parser.add_argument("--grid_id", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "init":
        run_initial_load()
    elif args.mode == "update":
        if args.grid_id:
            run_grid_update(args.grid_id)
        else:
            run_daily_update()
    elif args.mode == "status":
        print_db_status()

"""
data/data_store.py
──────────────────
DuckDB-backed analytical data store — Grid Edition.

Key changes from v2:
  - city column → grid_id (spatial key)
  - Added soil_moisture + lai_high_veg columns (8-feature schema)
  - shap_importance JSONB column in anomaly_scores for inline explainability
  - pincode_lookup table for geocoding
"""
import os
import sys
import json
from typing import Optional, List

import pandas as pd
import duckdb
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH, GRID_POINTS


# ─── Schema ──────────────────────────────────────────────────────────────────
CREATE_CLIMATE_TABLE = """
CREATE TABLE IF NOT EXISTS climate_daily (
    date             DATE        NOT NULL,
    grid_id          VARCHAR     NOT NULL,
    lat              DOUBLE,
    lon              DOUBLE,
    temp_max         DOUBLE,
    temp_min         DOUBLE,
    temp_mean        DOUBLE,
    precipitation    DOUBLE,
    wind_speed       DOUBLE,
    solar_radiation  DOUBLE,
    soil_moisture    DOUBLE,
    lai_high_veg     DOUBLE,
    is_mock          BOOLEAN     DEFAULT FALSE,
    PRIMARY KEY (date, grid_id)
);
"""

CREATE_ANOMALY_TABLE = """
CREATE TABLE IF NOT EXISTS anomaly_scores (
    date              DATE    NOT NULL,
    grid_id           VARCHAR NOT NULL,
    recon_error       DOUBLE,
    anomaly_score     DOUBLE,
    is_anomaly        BOOLEAN,
    anomaly_type      VARCHAR,
    severity          VARCHAR,
    shap_importance   VARCHAR,   -- JSON string: {"feature": importance_score, ...}
    briefing          VARCHAR,   -- Natural language summary
    PRIMARY KEY (date, grid_id)
);
"""

CREATE_FORECAST_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    generated_at           TIMESTAMP NOT NULL,
    grid_id                VARCHAR   NOT NULL,
    forecast_date          DATE      NOT NULL,
    temp_mean_pred         DOUBLE,
    temp_max_pred          DOUBLE,
    temp_min_pred          DOUBLE,
    precipitation_pred     DOUBLE,
    temp_mean_lower_95     DOUBLE,
    temp_mean_upper_95     DOUBLE,
    PRIMARY KEY (generated_at, grid_id, forecast_date)
);
"""

CREATE_PINCODE_TABLE = """
CREATE TABLE IF NOT EXISTS pincode_lookup (
    pincode    VARCHAR PRIMARY KEY,
    area_name  VARCHAR,
    district   VARCHAR,
    state      VARCHAR,
    lat        DOUBLE,
    lon        DOUBLE
);
"""


def get_connection() -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute(CREATE_CLIMATE_TABLE)
    conn.execute(CREATE_ANOMALY_TABLE)
    conn.execute(CREATE_FORECAST_TABLE)
    conn.execute(CREATE_PINCODE_TABLE)
    conn.commit()


# Columns that map to the DB schema
_CLIMATE_DB_COLS = [
    "date", "grid_id", "lat", "lon",
    "temp_max", "temp_min", "temp_mean",
    "precipitation", "wind_speed", "solar_radiation",
    "soil_moisture", "lai_high_veg", "is_mock",
]


def upsert_climate_data(df: pd.DataFrame) -> int:
    """Upsert 8-feature climate daily records into the store."""
    conn = get_connection()
    required = ["date", "grid_id", "temp_max", "temp_min", "temp_mean", "precipitation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    insert_df = df.copy()
    if "is_mock" not in insert_df.columns:
        insert_df["is_mock"] = False
    # Fill new feature columns with 0 if absent (backward compat)
    for col in ["soil_moisture", "lai_high_veg"]:
        if col not in insert_df.columns:
            insert_df[col] = 0.0
    insert_cols = [c for c in _CLIMATE_DB_COLS if c in insert_df.columns]
    insert_df = insert_df[insert_cols]

    conn.register("staging", insert_df)
    conn.execute("""
        INSERT OR REPLACE INTO climate_daily
        SELECT * FROM staging
    """)
    conn.commit()
    rows = len(insert_df)
    logger.info(f"Upserted {rows} climate records.")
    return rows


def upsert_anomaly_scores(df: pd.DataFrame) -> int:
    """Upsert anomaly scores including SHAP importance JSON and briefing text."""
    conn = get_connection()

    insert_df = df.copy()
    # Serialize shap dict to JSON if present
    if "shap_importance" in insert_df.columns:
        insert_df["shap_importance"] = insert_df["shap_importance"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else (x or "{}")
        )
    else:
        insert_df["shap_importance"] = "{}"

    if "briefing" not in insert_df.columns:
        insert_df["briefing"] = ""

    conn.register("staging_anomaly", insert_df)
    conn.execute("""
        INSERT OR REPLACE INTO anomaly_scores
        SELECT * FROM staging_anomaly
    """)
    conn.commit()
    rows = len(insert_df)
    logger.info(f"Upserted {rows} anomaly score records.")
    return rows


def upsert_forecasts(df: pd.DataFrame) -> int:
    conn = get_connection()
    conn.register("staging_forecast", df)
    conn.execute("""
        INSERT OR REPLACE INTO forecasts
        SELECT * FROM staging_forecast
    """)
    conn.commit()
    rows = len(df)
    logger.info(f"Upserted {rows} forecast records.")
    return rows


# ─── Read Operations ─────────────────────────────────────────────────────────

def get_climate_data(
    grid_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM climate_daily WHERE grid_id = ?"
    params = [grid_id]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    return conn.execute(query, params).df()


def get_anomaly_scores(grid_id: str, start_date: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM anomaly_scores WHERE grid_id = ?"
    params = [grid_id]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    query += " ORDER BY date"
    df = conn.execute(query, params).df()
    # Deserialize shap JSON back to dict
    if "shap_importance" in df.columns:
        df["shap_importance"] = df["shap_importance"].apply(
            lambda x: json.loads(x) if x and x != "{}" else {}
        )
    return df


def get_latest_forecast(grid_id: str) -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT * FROM forecasts
        WHERE grid_id = ?
          AND generated_at = (
              SELECT MAX(generated_at) FROM forecasts WHERE grid_id = ?
          )
        ORDER BY forecast_date
    """
    return conn.execute(query, [grid_id, grid_id]).df()


def get_all_grid_latest_status() -> pd.DataFrame:
    """Summary table used by the dashboard overview map — grid edition."""
    conn = get_connection()
    query = """
        SELECT
            c.grid_id,
            c.lat,
            c.lon,
            c.temp_mean        AS latest_temp,
            c.precipitation    AS latest_precip,
            c.soil_moisture,
            c.lai_high_veg,
            a.anomaly_score,
            a.is_anomaly,
            a.severity,
            a.anomaly_type,
            a.shap_importance,
            a.briefing
        FROM climate_daily c
        LEFT JOIN anomaly_scores a
            ON c.date = a.date AND c.grid_id = a.grid_id
        WHERE c.date = (SELECT MAX(date) FROM climate_daily)
          AND (a.date IS NULL OR a.date = (SELECT MAX(date) FROM anomaly_scores))
    """
    df = conn.execute(query).df()
    if "shap_importance" in df.columns:
        df["shap_importance"] = df["shap_importance"].apply(
            lambda x: json.loads(x) if x and x != "{}" else {}
        )
    return df


def lookup_pincode(pincode: str) -> Optional[dict]:
    """Return lat/lon for a pincode if it exists in the lookup table."""
    conn = get_connection()
    result = conn.execute(
        "SELECT * FROM pincode_lookup WHERE pincode = ?", [str(pincode).strip()]
    ).df()
    if result.empty:
        return None
    return result.iloc[0].to_dict()


def upsert_pincode_data(df: pd.DataFrame) -> int:
    """Bulk-load pincode → lat/lon mapping."""
    conn = get_connection()
    conn.register("staging_pincode", df)
    conn.execute("""
        INSERT OR REPLACE INTO pincode_lookup
        SELECT pincode, area_name, district, state, lat, lon
        FROM staging_pincode
    """)
    conn.commit()
    rows = len(df)
    logger.info(f"Upserted {rows} pincode records.")
    return rows


def get_anomaly_summary(start: str) -> list:
    """Per-grid anomaly summary for the last N days."""
    conn = get_connection()
    query = """
        SELECT
            grid_id,
            lat, lon,
            COUNT(*) FILTER (WHERE is_anomaly = TRUE)    AS anomaly_days,
            COUNT(*) FILTER (WHERE severity = 'extreme') AS extreme_days,
            MAX(anomaly_score)                           AS peak_score,
            MAX(briefing)                                AS latest_briefing
        FROM anomaly_scores
        WHERE date >= ?
        GROUP BY grid_id, lat, lon
    """
    try:
        return conn.execute(query, [start]).df().to_dict(orient="records")
    except Exception:
        return []


def get_db_stats() -> dict:
    conn = get_connection()
    stats = {}
    stats["climate_rows"]  = conn.execute("SELECT COUNT(*) FROM climate_daily").fetchone()[0]
    stats["anomaly_rows"]  = conn.execute("SELECT COUNT(*) FROM anomaly_scores").fetchone()[0]
    stats["forecast_rows"] = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    stats["grid_ids"]      = conn.execute("SELECT DISTINCT grid_id FROM climate_daily").df()["grid_id"].tolist()
    stats["date_range"]    = conn.execute("SELECT MIN(date), MAX(date) FROM climate_daily").fetchone()
    return stats

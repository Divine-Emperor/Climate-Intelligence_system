"""
api/routes/data.py
──────────────────
REST endpoints for raw climate data access — Grid Edition.

New endpoints:
  GET  /grid_points      — list all grid points
  GET  /climate/{grid_id} — raw climate data for a grid point
  GET  /overview         — latest status across ALL grid points
  POST /geocode          — convert pincode/location to nearest grid_id
  GET  /stats            — DB statistics
"""
import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import GRID_POINTS, GRID_COORDS, GRID_IDS
from data.data_store import (
    get_climate_data,
    get_all_grid_latest_status,
    get_db_stats,
    lookup_pincode,
)

router = APIRouter()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _nearest_grid_id(lat: float, lon: float) -> dict:
    """Find the closest grid point to the given lat/lon using L2 distance."""
    import numpy as np
    dists   = np.sqrt((GRID_COORDS[:, 0] - lat) ** 2 + (GRID_COORDS[:, 1] - lon) ** 2)
    idx     = int(np.argmin(dists))
    grid_id = GRID_IDS[idx]
    dist_km = float(dists[idx] * 111)   # ~111km per degree
    info    = GRID_POINTS[grid_id]
    return {
        "grid_id":    grid_id,
        "lat":        info["lat"],
        "lon":        info["lon"],
        "dist_km":    round(dist_km, 1),
    }


def _geocode_text(location: str) -> Optional[dict]:
    """Try to geocode a location string using geopy (Nominatim)."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geocoder = Nominatim(user_agent="climate_ai_v2", timeout=8)
        result   = geocoder.geocode(f"{location}, India")
        if result:
            return {"lat": result.latitude, "lon": result.longitude}
    except Exception as e:
        logger.warning(f"Geocoding '{location}' failed: {e}")
    return None


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class GeocodeRequest(BaseModel):
    query: str           # Pincode (e.g. "380001") or city name (e.g. "Ahmedabad")


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/grid_points")
def list_grid_points():
    """Return all configured grid points with coordinates."""
    return {
        "grid_points": [
            {"grid_id": gid, "lat": info["lat"], "lon": info["lon"]}
            for gid, info in GRID_POINTS.items()
        ],
        "total": len(GRID_POINTS),
    }


@router.get("/cities")
def list_cities():
    """Backward-compatible alias for /grid_points."""
    return list_grid_points()


@router.get("/climate/{grid_id}")
def get_grid_climate(
    grid_id: str,
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date:   Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Return raw 8-feature climate records for a grid point over a date range."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")
    df = get_climate_data(grid_id, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found. Run the data pipeline first.")
    return df.to_dict(orient="records")


@router.get("/overview")
def get_overview():
    """Latest-day status for all grid points — used by the dashboard overview map."""
    df = get_all_grid_latest_status()
    if df.empty:
        return []
    # Convert shap_importance dict to JSON-safe format
    if "shap_importance" in df.columns:
        df["shap_importance"] = df["shap_importance"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
    return df.to_dict(orient="records")


@router.post("/geocode")
def geocode_location(req: GeocodeRequest):
    """
    Convert a pincode or location query into the nearest grid point.

    Priority:
      1. Check local pincode_lookup table (instant, no network)
      2. Fall back to Nominatim geocoding (requires network)
      3. If both fail, return 404
    """
    query = req.query.strip()

    # 1. Pincode lookup (numeric → lookup table)
    if query.isdigit() and len(query) == 6:
        row = lookup_pincode(query)
        if row:
            nearest = _nearest_grid_id(row["lat"], row["lon"])
            return {
                "query":     query,
                "resolved":  f"{row.get('area_name', '')}, {row.get('district', '')}",
                "input_lat": row["lat"],
                "input_lon": row["lon"],
                **nearest,
            }

    # 2. Text geocoding via Nominatim
    coords = _geocode_text(query)
    if coords:
        nearest = _nearest_grid_id(coords["lat"], coords["lon"])
        return {
            "query":     query,
            "resolved":  query,
            "input_lat": coords["lat"],
            "input_lon": coords["lon"],
            **nearest,
        }

    raise HTTPException(
        status_code=404,
        detail=f"Could not resolve '{query}' to coordinates. "
               "Try a 6-digit Indian pincode or city name."
    )


@router.get("/stats")
def get_stats():
    """Return database statistics."""
    return get_db_stats()

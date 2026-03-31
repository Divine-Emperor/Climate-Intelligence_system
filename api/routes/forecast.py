"""
api/routes/forecast.py
──────────────────────
REST endpoints for climate forecasting — Grid Edition.

Upgrades from v2:
  - Queries by grid_id instead of city
  - Fits with ensemble forecasting model output schema
"""
import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import GRID_POINTS
from data.data_store import get_climate_data, get_latest_forecast, upsert_forecasts
from models.forecaster import ClimateForecaster

router = APIRouter()

_forecasters: dict = {}


def _get_forecaster(grid_id: str) -> ClimateForecaster:
    if grid_id not in _forecasters:
        fc = ClimateForecaster(grid_id)
        fc.load()
        _forecasters[grid_id] = fc
    return _forecasters[grid_id]


@router.get("/latest/{grid_id}")
def get_latest_forecast_endpoint(grid_id: str):
    """Return the most recently generated forecast for a grid point."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")
    df = get_latest_forecast(grid_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecast found. Run /train and /generate first.")
    return df.to_dict(orient="records")


@router.post("/generate/{grid_id}")
def generate_forecast(grid_id: str, background_tasks: BackgroundTasks):
    """Generate a fresh 10-day forecast for a grid point using the trained Ensemble Attention-BiLSTM model."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")

    def _predict():
        try:
            df  = get_climate_data(grid_id)
            fc  = _get_forecaster(grid_id)
            out = fc.predict(df)
            upsert_forecasts(out)
            logger.info(f"Forecast generated for {grid_id}.")
        except Exception as e:
            logger.error(f"Forecast failed for {grid_id}: {e}")

    background_tasks.add_task(_predict)
    return {"message": f"Forecast generation started for {grid_id}."}


@router.post("/train/{grid_id}")
def train_forecast_model(grid_id: str, background_tasks: BackgroundTasks):
    """Train (or retrain) the Ensemble BiLSTM forecaster for a grid point."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")

    def _train():
        try:
            df = get_climate_data(grid_id)
            fc = ClimateForecaster(grid_id)
            fc.train(df)
            _forecasters[grid_id] = fc
            logger.info(f"Forecaster training complete for {grid_id}.")
        except Exception as e:
            logger.error(f"Forecaster training failed for {grid_id}: {e}")

    background_tasks.add_task(_train)
    return {"message": f"Forecaster training started for {grid_id}."}

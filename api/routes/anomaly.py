"""
api/routes/anomaly.py
─────────────────────
REST endpoints for anomaly detection — Grid Edition.

Upgrades from v2:
  - Queries by grid_id instead of city
  - POST /shap/{grid_id} triggers async SHAP computation
  - GET /summary returns unified briefing text
  - GET /briefing returns the AI-generated intelligence report
"""
import os
import sys
import json
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import GRID_POINTS, GRID_IDS
from data.data_store import (
    get_climate_data,
    get_anomaly_scores,
    upsert_anomaly_scores,
    get_anomaly_summary,
)
from models.anomaly_detector import SpatiotemporalAnomalyDetector
from models.shap_explainer import run_shap_for_grid, generate_anomaly_briefing

router = APIRouter()

_detectors: dict = {}


def _get_detector(grid_id: str) -> SpatiotemporalAnomalyDetector:
    if grid_id not in _detectors:
        det = SpatiotemporalAnomalyDetector(grid_id)
        det.load()
        _detectors[grid_id] = det
    return _detectors[grid_id]


# ─── Scoring ─────────────────────────────────────────────────────────────────

@router.get("/scores/{grid_id}")
def get_anomaly_scores_endpoint(
    grid_id: str,
    start_date: Optional[str] = Query(None),
):
    """Return pre-computed anomaly scores for a grid point."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")
    df = get_anomaly_scores(grid_id, start_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No anomaly scores found. Run /score first.")
    # Serialize shap dict to JSON-safe
    if "shap_importance" in df.columns:
        df["shap_importance"] = df["shap_importance"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
    return df.to_dict(orient="records")


@router.post("/score/{grid_id}")
def run_anomaly_scoring(grid_id: str, background_tasks: BackgroundTasks):
    """Trigger anomaly scoring for a grid point using the trained STAE model."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")

    def _score():
        try:
            df     = get_climate_data(grid_id)
            det    = _get_detector(grid_id)
            scores = det.score(df)
            # Add grid_id column if missing
            if "grid_id" not in scores.columns:
                scores.insert(1, "grid_id", grid_id)
            upsert_anomaly_scores(scores)
            logger.info(f"Anomaly scoring complete for {grid_id}.")
        except Exception as e:
            logger.error(f"Scoring failed for {grid_id}: {e}")

    background_tasks.add_task(_score)
    return {"message": f"Anomaly scoring started for {grid_id}."}


# ─── Training ────────────────────────────────────────────────────────────────

@router.post("/train/{grid_id}")
def train_anomaly_model(grid_id: str, background_tasks: BackgroundTasks):
    """Train (or retrain) the STAE model for a grid point."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")

    def _train():
        try:
            df  = get_climate_data(grid_id)
            det = SpatiotemporalAnomalyDetector(grid_id)
            det.train(df)
            _detectors[grid_id] = det
            logger.info(f"STAE training complete for {grid_id}.")
        except Exception as e:
            logger.error(f"Training failed for {grid_id}: {e}")

    background_tasks.add_task(_train)
    return {"message": f"STAE model training started for {grid_id}."}


# ─── SHAP ─────────────────────────────────────────────────────────────────────

@router.post("/shap/{grid_id}")
def compute_shap(grid_id: str, background_tasks: BackgroundTasks):
    """Trigger async SHAP importance computation for a grid point."""
    if grid_id not in GRID_POINTS:
        raise HTTPException(status_code=404, detail=f"Grid point '{grid_id}' not found.")

    def _run_shap():
        try:
            det = _get_detector(grid_id)
            if det.model is None:
                logger.warning(f"[{grid_id}] No trained model — skipping SHAP.")
                return

            df         = get_climate_data(grid_id)
            anomaly_df = get_anomaly_scores(grid_id)
            if anomaly_df.empty:
                return

            shap_results = run_shap_for_grid(
                grid_id=grid_id,
                stae_model=det.model,
                scaler=det.scaler,
                df=df,
                anomaly_df=anomaly_df,
            )

            # Merge SHAP results back into anomaly_scores table
            if shap_results:
                for date_key, payload in shap_results.items():
                    mask = anomaly_df["date"].astype(str) == date_key
                    if mask.any():
                        anomaly_df.loc[mask, "shap_importance"] = json.dumps(payload["shap_importance"])
                        anomaly_df.loc[mask, "briefing"]        = payload["briefing"]
                upsert_anomaly_scores(anomaly_df)
                logger.info(f"[{grid_id}] SHAP results saved for {len(shap_results)} anomaly days.")
        except Exception as e:
            logger.error(f"SHAP computation failed for {grid_id}: {e}")

    background_tasks.add_task(_run_shap)
    return {"message": f"SHAP computation started for {grid_id}."}


# ─── Summary & Briefing ───────────────────────────────────────────────────────

@router.get("/summary")
def get_anomaly_summary_endpoint(days: int = Query(30, ge=1, le=365)):
    """High-level summary: anomaly counts per grid point for the last N days."""
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return get_anomaly_summary(start)


@router.get("/briefing")
def get_unified_briefing(days: int = Query(30, ge=1, le=90)):
    """
    Return the full unified AI anomaly intelligence briefing —
    synthesized natural-language paragraphs for all active anomaly grid points.
    """
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    summary = get_anomaly_summary(start)

    if not summary:
        return {"briefing": "✅ No active anomalies detected across the monitoring grid.", "count": 0}

    lines = [f"🌐 Climate Anomaly Intelligence Report — Last {days} Days\n"]
    active = [s for s in summary if s.get("anomaly_days", 0) > 0]
    stable = [s for s in summary if s.get("anomaly_days", 0) == 0]

    # Sort by severity (extreme first)
    active.sort(key=lambda x: -(x.get("extreme_days", 0) * 10 + x.get("anomaly_days", 0)))

    for s in active:
        briefing = s.get("latest_briefing") or ""
        if briefing:
            lines.append(briefing)
        else:
            gid     = s["grid_id"]
            n_days  = s["anomaly_days"]
            extreme = s.get("extreme_days", 0)
            icon    = "🔴" if extreme > 2 else ("🟡" if n_days > 5 else "🟠")
            lines.append(f"{icon} {gid}: {n_days} anomaly days "
                         + (f"({extreme} extreme)" if extreme else ""))

    if stable:
        lines.append(f"\n✅ {len(stable)} grid points within normal bounds.")

    return {
        "briefing": "\n".join(lines),
        "active_count": len(active),
        "stable_count": len(stable),
    }

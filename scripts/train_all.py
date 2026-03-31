"""
scripts/train_all.py
─────────────────────
One-shot CLI training script (Grid Edition).
Loads 8-feature data from DuckDB, trains both STAE anomaly detector and 
Attention-BiLSTM forecaster for all grid points, computes SHAP, logs to MLflow.

Usage:
  python scripts/train_all.py                     # Train all grid points
  python scripts/train_all.py --grid_id Grid_22.0_76.0  # Train a single point
  python scripts/train_all.py --model anomaly     # Only train anomaly model
  python scripts/train_all.py --model forecast    # Only train forecast model
"""
import os
import sys
import argparse
import time

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRID_POINTS, LOG_FILE, LOG_LEVEL, SEQ_LENGTH, EPOCHS, BATCH_SIZE
from data.data_store import get_climate_data, get_db_stats, upsert_anomaly_scores
from data.pipeline import run_initial_load
from models.anomaly_detector import SpatiotemporalAnomalyDetector
from models.forecaster import ClimateForecaster
from models.shap_explainer import run_shap_for_grid
from models.model_registry import log_training_run, setup_mlflow

os.makedirs("logs", exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level=LOG_LEVEL)


def ensure_data():
    """Check DB has data; if not, run initial pipeline load."""
    stats = get_db_stats()
    if stats["climate_rows"] == 0:
        logger.warning("Database is empty — running initial data load from GEE ...")
        run_initial_load()
    else:
        logger.info(
            f"DB ready: {stats['climate_rows']:,} climate records across {len(stats.get('grid_ids', []))} grid points."
        )


def train_anomaly(grid_id: str):
    logger.info(f"━━━ Training STAE Anomaly Detector for {grid_id} ━━━")
    df = get_climate_data(grid_id)
    if len(df) < SEQ_LENGTH * 3:
        logger.warning(f"Not enough data for {grid_id} (need at least {SEQ_LENGTH * 3} rows). Skipping.")
        return

    t0  = time.time()
    det = SpatiotemporalAnomalyDetector(grid_id)
    det.train(df)

    # Score
    scores = det.score(df)
    
    elapsed = time.time() - t0
    n_anom  = int(scores["is_anomaly"].sum()) if "is_anomaly" in scores.columns else 0

    # Compute SHAP baseline for explainability
    if n_anom > 0:
        logger.info(f"[{grid_id}] Computing SHAP importances for {n_anom} anomaly days...")
        import json
        shap_results = run_shap_for_grid(grid_id, det.model, det.scaler, df, scores)
        if shap_results:
            for date_key, payload in shap_results.items():
                mask = scores["date"].astype(str) == date_key
                if mask.any():
                    scores.loc[mask, "shap_importance"] = json.dumps(payload["shap_importance"])
                    scores.loc[mask, "briefing"]        = payload["briefing"]
                    logger.debug(f"Attached SHAP values & briefing to {date_key}")

    upsert_anomaly_scores(scores)
    
    try:
        log_training_run(
            model_name="stae_anomaly",
            city=grid_id,
            params={"seq_length": SEQ_LENGTH, "epochs": EPOCHS, "batch_size": BATCH_SIZE},
            metrics={"training_time_s": round(elapsed, 1), "anomaly_days_detected": n_anom,
                     "threshold": float(det.threshold)},
            model=det.model,
        )
    except Exception as e:
        logger.warning(f"MLFlow logging failed: {e}")

    logger.success(f"[{grid_id}] STAE done in {elapsed:.0f}s | {n_anom} anomalies detected.")


def train_forecast(grid_id: str):
    logger.info(f"━━━ Training Attention-BiLSTM Forecaster for {grid_id} ━━━")
    df = get_climate_data(grid_id)
    if len(df) < SEQ_LENGTH * 3:
        logger.warning(f"Not enough data for {grid_id}. Skipping.")
        return

    t0 = time.time()
    fc = ClimateForecaster(grid_id)
    fc.train(df)
    elapsed = time.time() - t0

    try:
        log_training_run(
            model_name="attention_bilstm_forecast",
            city=grid_id,
            params={"seq_length": SEQ_LENGTH, "epochs": EPOCHS, "batch_size": BATCH_SIZE},
            metrics={"training_time_s": round(elapsed, 1)},
            model=fc.models[0] if fc.models else None,
        )
    except Exception as e:
        logger.warning(f"MLFlow logging failed: {e}")

    logger.success(f"[{grid_id}] Forecaster done in {elapsed:.0f}s.")


def main():
    parser = argparse.ArgumentParser(description="Climate AI Grid — Batch Model Training")
    parser.add_argument("--grid_id", type=str, default=None,
                        help="Grid ID to train. Default: all grid points.")
    parser.add_argument("--city",    type=str, default=None,
                        help="Legacy alias for --grid_id")
    parser.add_argument("--model",   type=str, default="all",
                        choices=["all", "anomaly", "forecast"],
                        help="Which model to train.")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip DB data check (useful in CI).")
    args = parser.parse_args()

    setup_mlflow()

    if not args.skip_data_check:
        ensure_data()

    target = args.grid_id or args.city
    targets = [target] if target else list(GRID_POINTS.keys())
    logger.info(f"Training targets: {len(targets)} | Model: {args.model}")

    for target_id in targets:
        try:
            if args.model in ("all", "anomaly"):
                train_anomaly(target_id)
            if args.model in ("all", "forecast"):
                train_forecast(target_id)
        except Exception as e:
            logger.error(f"Training failed for {target_id}: {e}")
            continue

    logger.success("━━━ All training complete. ━━━")


if __name__ == "__main__":
    main()

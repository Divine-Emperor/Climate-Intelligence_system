"""
scripts/scheduler.py
─────────────────────
APScheduler-based background daemon.
Runs two recurring jobs:
  1. Daily data update — fetches latest GEE climate data at 01:00 UTC
  2. Weekly retraining — retrains all models every Sunday at 03:00 UTC

Start via:  python scripts/scheduler.py
"""
import os
import sys
import signal
import time

from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FOCAL_CITIES, LOG_FILE, LOG_LEVEL
from data.pipeline import run_daily_update
from data.data_store import upsert_anomaly_scores, upsert_forecasts

os.makedirs("logs", exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level=LOG_LEVEL)

scheduler = BlockingScheduler(timezone="UTC")


# ─── Job Definitions ─────────────────────────────────────────────────────────

def job_daily_update():
    """Fetch latest climate data + score anomalies + generate forecasts."""
    logger.info("⏰ [Scheduler] Daily update job triggered.")
    try:
        run_daily_update()
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        return

    # Re-score anomalies for all cities
    from models.anomaly_detector import SpatiotemporalAnomalyDetector
    from models.forecaster import ClimateForecaster
    from data.data_store import get_climate_data
    from datetime import datetime, timedelta

    start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    for city in FOCAL_CITIES:
        try:
            df = get_climate_data(city)
            if df.empty:
                continue

            det = SpatiotemporalAnomalyDetector(city)
            if det.load():
                scores = det.score(df[df["date"] >= start] if "date" in df.columns else df)
                upsert_anomaly_scores(scores)
                logger.info(f"[{city}] Anomaly scores updated.")

            fc = ClimateForecaster(city)
            if fc.load():
                forecast = fc.predict(df)
                upsert_forecasts(forecast)
                logger.info(f"[{city}] Forecast updated.")
        except Exception as e:
            logger.error(f"Post-update scoring failed for {city}: {e}")

    logger.success("⏰ [Scheduler] Daily update complete.")


def job_weekly_retrain():
    """Retrain all models with fresh data every Sunday."""
    logger.info("⏰ [Scheduler] Weekly retraining triggered.")
    try:
        # Delegate to the train_all script logic
        from scripts.train_all import train_anomaly, train_forecast, ensure_data
        ensure_data()
        for city in FOCAL_CITIES:
            try:
                train_anomaly(city)
                train_forecast(city)
            except Exception as e:
                logger.error(f"Retraining failed for {city}: {e}")
        logger.success("⏰ [Scheduler] Weekly retraining complete.")
    except Exception as e:
        logger.error(f"Weekly retraining job failed: {e}")


# ─── Schedule Registration ────────────────────────────────────────────────────

scheduler.add_job(
    job_daily_update,
    CronTrigger(hour=1, minute=0),   # 01:00 UTC daily
    id="daily_update",
    name="Daily GEE Data Update + Scoring",
    max_instances=1,
    misfire_grace_time=3600,
)

scheduler.add_job(
    job_weekly_retrain,
    CronTrigger(day_of_week="sun", hour=3, minute=0),  # Sunday 03:00 UTC
    id="weekly_retrain",
    name="Weekly Model Retraining",
    max_instances=1,
    misfire_grace_time=7200,
)


def _graceful_shutdown(signum, frame):
    logger.info("Received shutdown signal — stopping scheduler.")
    scheduler.shutdown(wait=False)
    sys.exit(0)


signal.signal(signal.SIGINT,  _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)


if __name__ == "__main__":
    logger.info("🕐 Climate AI Scheduler starting ...")
    logger.info("  • Daily update   : 01:00 UTC every day")
    logger.info("  • Weekly retrain : 03:00 UTC every Sunday")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")

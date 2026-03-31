"""
api/main.py
───────────
FastAPI application root.
Mounts all route groups and handles startup/shutdown lifecycle.
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_HOST, API_PORT, LOG_FILE, LOG_LEVEL
from api.routes import anomaly, forecast, data as data_route

os.makedirs("logs", exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level=LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Climate AI API starting up …")
    yield
    logger.info("Climate AI API shut down.")


app = FastAPI(
    title="Climate AI — Spatiotemporal Anomaly Detection API",
    description=(
        "Production-grade API for detecting and forecasting climate anomalies "
        "across Indian focal cities using GEE satellite data and deep learning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(data_route.router,    prefix="/api/v1/data",     tags=["Data"])
app.include_router(anomaly.router,       prefix="/api/v1/anomaly",  tags=["Anomaly"])
app.include_router(forecast.router,      prefix="/api/v1/forecast", tags=["Forecast"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "Climate AI v2", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)

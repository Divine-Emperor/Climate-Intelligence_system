# 🌍 Climate AI v2 — Spatiotemporal Climate Anomaly Detection

> **Problem Statement (Environmental & Climate Intelligence #3)**
> Extreme climate events such as heatwaves and abnormal rainfall are increasing in frequency. Build an AI-based spatiotemporal system capable of detecting and forecasting climate anomalies using geospatial time-series and satellite data.

---

## Architecture Overview

```
┌─────────────────┐     REST API      ┌────────────────────┐
│  Streamlit UI   │ ◄──────────────── │   FastAPI Backend  │
│  (Port 8501)    │                   │   (Port 8000)      │
└─────────────────┘                   └────────┬───────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                   ┌──────▼──────┐   ┌─────────▼────────┐  ┌───────▼──────┐
                   │   DuckDB    │   │  STAE Anomaly    │  │ BiLSTM       │
                   │  Data Store │   │  Detector        │  │ Forecaster   │
                   └──────▲──────┘   └──────────────────┘  └──────────────┘
                          │
                   ┌──────┴──────┐
                   │ GEE Data    │
                   │ Pipeline    │  ← ERA5 / MODIS / GPM
                   └─────────────┘
```

## Project Structure

```
climate-ai-v2/
├── config.py                  # Central config: GEE, cities, model params
├── requirements.txt           # All Python dependencies
├── run.sh                     # Local launch script
├── docker-compose.yml         # Full system orchestration
├── Dockerfile.api             # FastAPI container
├── Dockerfile.frontend        # Streamlit container
├── .env.example               # Environment variable template
│
├── data/
│   ├── gee_collector.py       # Google Earth Engine ERA5 ingestion
│   ├── data_store.py          # DuckDB analytical database
│   └── pipeline.py            # Fetch → Validate → Store orchestrator
│
├── models/
│   ├── anomaly_detector.py    # Spatiotemporal Autoencoder (STAE)
│   ├── forecaster.py          # Multivariate BiLSTM + MC Dropout
│   └── model_registry.py     # MLflow experiment tracking
│
├── api/
│   ├── main.py                # FastAPI app root
│   └── routes/
│       ├── data.py            # /api/v1/data/*
│       ├── anomaly.py         # /api/v1/anomaly/*
│       └── forecast.py        # /api/v1/forecast/*
│
├── frontend/
│   ├── app.py                 # Premium Streamlit dashboard
│   └── components/
│       ├── risk_map.py        # PyDeck interactive risk map
│       └── charts.py          # Plotly temporal charts
│
├── scripts/
│   ├── train_all.py           # Batch model training CLI
│   └── scheduler.py           # APScheduler daily/weekly automation
│
└── tests/
    ├── test_data_store.py     # Data layer unit tests
    ├── test_anomaly_detector.py  # STAE unit tests
    └── test_api.py            # FastAPI integration tests
```

## Quick Start

### 1. Setup

```bash
# Clone / navigate to the project
cd "climate-ai-v2"

# Copy and fill in environment variables
cp .env.example .env
# Edit .env → set GEE_SERVICE_ACCOUNT, GEE_KEY_FILE
```

### 2. Add your GEE key

Place your GEE service account JSON key file at the project root as `gee-key.json`.

```bash
# Verify GEE auth works
python3 -c "import ee; ee.Initialize(); print('GEE OK')"
```

### 3. Load data & train models (first run only)

```bash
./run.sh --init
# This will:
#   1. Fetch historical ERA5 data (2018–2024) for all 8 Indian cities from GEE
#   2. Store in DuckDB
#   3. Train STAE + BiLSTM for each city
#   4. Start API + Frontend
```

### 4. Regular startup

```bash
./run.sh          # Start API (port 8000) + Frontend (port 8501)
```

### 5. Docker (production)

```bash
# Copy your GEE key and set env
cp gee-key.json ./gee-key.json

# Start everything
docker-compose up --build

# Services:
#   Frontend  → http://localhost:8501
#   API       → http://localhost:8000
#   API Docs  → http://localhost:8000/docs
#   MLflow    → http://localhost:5000
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/data/cities` | List all focal cities |
| GET | `/api/v1/data/climate/{city}` | Get climate time-series |
| GET | `/api/v1/data/overview` | Latest status for all cities (map data) |
| GET | `/api/v1/anomaly/scores/{city}` | Pre-computed anomaly scores |
| POST | `/api/v1/anomaly/train/{city}` | Train STAE for a city |
| POST | `/api/v1/anomaly/score/{city}` | Run scoring + persist results |
| GET | `/api/v1/anomaly/summary` | 30-day anomaly count per city |
| GET | `/api/v1/forecast/latest/{city}` | Latest 10-day forecast |
| POST | `/api/v1/forecast/train/{city}` | Train BiLSTM forecaster |
| POST | `/api/v1/forecast/generate/{city}` | Generate new forecast |

Full Swagger docs: `http://localhost:8000/docs`

## ML Models

### STAE — Spatiotemporal Autoencoder
- **Input**: 30-day sliding window of 6 climate variables
- **Architecture**: Bidirectional LSTM Encoder → Bottleneck → Bidirectional LSTM Decoder
- **Anomaly Detection**: Days where reconstruction MAE > 65th percentile threshold
- **Output**: Anomaly score (0–1), type (heatwave/heavy_rain/cold_spell/compound), severity

### BiLSTM Forecaster
- **Input**: Last 30 days of 6 climate variables
- **Output**: 10-day forward forecast for temp_max, temp_min, temp_mean, precipitation
- **Uncertainty**: Monte Carlo Dropout (30 stochastic passes → 95% confidence intervals)

## Focal Cities (India)

| City | Lat | Lon |
|------|-----|-----|
| Delhi | 28.61 | 77.21 |
| Mumbai | 19.08 | 72.88 |
| Chennai | 13.08 | 80.27 |
| Kolkata | 22.57 | 88.36 |
| Bengaluru | 12.97 | 77.59 |
| Hyderabad | 17.39 | 78.49 |
| Ahmedabad | 23.02 | 72.57 |
| Jaipur | 26.91 | 75.79 |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Scheduler (Automated Operations)

```bash
# Start background scheduler (runs as a daemon)
python scripts/scheduler.py

# Schedule:
#   01:00 UTC daily  — fetch new GEE data, re-score anomalies, generate forecasts
#   03:00 UTC Sunday — retrain all models with fresh data
```

## Data Sources

| Source | Dataset | Variables |
|--------|---------|-----------|
| Google Earth Engine | ERA5 Land Daily | temp_max, temp_min, precipitation, wind, solar radiation |
| GEE | MODIS MOD11A1 | Land surface temperature |
| GEE | NASA GPM IMERG | Precipitation |

---


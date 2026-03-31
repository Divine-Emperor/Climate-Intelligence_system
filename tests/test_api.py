"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI API routes.
Uses httpx TestClient (no server process needed).
DB is pointed to a temp file so tests are isolated.
"""
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point to a clean temp database before any imports touch it
_tmp_db = os.path.join(tempfile.mkdtemp(), "test_api.duckdb")
os.environ["DB_PATH"] = _tmp_db

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ─── Seed data fixture ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed_db():
    """Seed the test DB with a minimal Delhi climate dataset."""
    from data.data_store import upsert_climate_data, upsert_anomaly_scores
    np.random.seed(0)
    n = 90
    dates = pd.date_range("2023-01-01", periods=n)
    climate_df = pd.DataFrame({
        "date":            dates,
        "city":            "Delhi",
        "lat":             28.6139,
        "lon":             77.2090,
        "temp_max":        35 + np.random.normal(0, 3, n),
        "temp_min":        22 + np.random.normal(0, 2, n),
        "temp_mean":       28 + np.random.normal(0, 2, n),
        "precipitation":   np.maximum(0, np.random.exponential(2, n)),
        "wind_speed":      3.0,
        "solar_radiation": 15e6,
        "is_mock":         True,
    })
    upsert_climate_data(climate_df)

    anomaly_df = pd.DataFrame({
        "date":          dates,
        "city":          "Delhi",
        "recon_error":   [0.02] * 88 + [0.16, 0.20],
        "anomaly_score": [0.15] * 88 + [0.72, 0.91],
        "is_anomaly":    [False] * 88 + [True, True],
        "anomaly_type":  ["none"] * 88 + ["heatwave", "heatwave"],
        "severity":      ["low"] * 88 + ["high", "extreme"],
    })
    upsert_anomaly_scores(anomaly_df)


# ─── Health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200


# ─── Data Routes ─────────────────────────────────────────────────────────────

class TestDataRoutes:
    def test_list_cities(self):
        r = client.get("/api/v1/data/cities")
        assert r.status_code == 200
        cities = [c["name"] for c in r.json()["cities"]]
        assert "Delhi" in cities
        assert "Mumbai" in cities

    def test_get_city_climate(self):
        r = client.get("/api/v1/data/climate/Delhi")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 90
        assert "temp_mean" in data[0]

    def test_date_filter(self):
        r = client.get("/api/v1/data/climate/Delhi",
                       params={"start_date": "2023-01-01", "end_date": "2023-01-31"})
        assert r.status_code == 200
        assert len(r.json()) == 31

    def test_unknown_city_404(self):
        r = client.get("/api/v1/data/climate/Atlantis")
        assert r.status_code == 404

    def test_stats(self):
        r = client.get("/api/v1/data/stats")
        assert r.status_code == 200
        stats = r.json()
        assert stats["climate_rows"] >= 90


# ─── Anomaly Routes ───────────────────────────────────────────────────────────

class TestAnomalyRoutes:
    def test_get_scores(self):
        r = client.get("/api/v1/anomaly/scores/Delhi")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 90
        is_anomaly_vals = [d["is_anomaly"] for d in data]
        assert True in is_anomaly_vals

    def test_get_scores_unknown_city(self):
        r = client.get("/api/v1/anomaly/scores/Atlantis")
        assert r.status_code == 404

    def test_score_trigger(self):
        # Note: scoring will fail (no trained model) but the endpoint itself should accept the request
        r = client.post("/api/v1/anomaly/score/Delhi")
        assert r.status_code == 200
        assert "started" in r.json()["message"].lower()

    def test_summary(self):
        r = client.get("/api/v1/anomaly/summary")
        assert r.status_code == 200
        results = r.json()
        delhi_entry = next((x for x in results if x["city"] == "Delhi"), None)
        assert delhi_entry is not None
        assert delhi_entry["anomaly_days"] == 2


# ─── Forecast Routes ──────────────────────────────────────────────────────────

class TestForecastRoutes:
    def test_forecast_not_found(self):
        r = client.get("/api/v1/forecast/latest/Delhi")
        # No forecast generated yet → 404
        assert r.status_code == 404

    def test_generate_trigger(self):
        r = client.post("/api/v1/forecast/generate/Delhi")
        assert r.status_code == 200
        assert "started" in r.json()["message"].lower()

    def test_train_trigger(self):
        r = client.post("/api/v1/forecast/train/Delhi")
        assert r.status_code == 200

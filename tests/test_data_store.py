"""
tests/test_data_store.py
─────────────────────────
Unit tests for the DuckDB data store.
"""
import os
import sys
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override DB path to an in-memory / temp DB for tests
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_climate.duckdb")

from data.data_store import (
    upsert_climate_data,
    get_climate_data,
    upsert_anomaly_scores,
    get_anomaly_scores,
    upsert_forecasts,
    get_latest_forecast,
    get_db_stats,
)


@pytest.fixture
def sample_climate_df():
    return pd.DataFrame({
        "date":            pd.date_range("2023-01-01", periods=60),
        "city":            "Delhi",
        "lat":             28.6139,
        "lon":             77.2090,
        "temp_max":        30.0 + pd.Series(range(60)) * 0.1,
        "temp_min":        20.0 + pd.Series(range(60)) * 0.05,
        "temp_mean":       25.0 + pd.Series(range(60)) * 0.07,
        "precipitation":   pd.Series(range(60)) % 5 * 2.0,
        "wind_speed":      3.5,
        "solar_radiation": 15e6,
        "is_mock":         True,
    })


@pytest.fixture
def sample_anomaly_df():
    return pd.DataFrame({
        "date":          pd.date_range("2023-01-31", periods=30),
        "city":          "Delhi",
        "recon_error":   [0.02] * 28 + [0.15, 0.18],
        "anomaly_score": [0.1] * 28 + [0.75, 0.85],
        "is_anomaly":    [False] * 28 + [True, True],
        "anomaly_type":  ["none"] * 28 + ["heatwave", "heatwave"],
        "severity":      ["low"] * 28 + ["high", "extreme"],
    })


@pytest.fixture
def sample_forecast_df():
    return pd.DataFrame({
        "generated_at":      pd.Timestamp.utcnow(),
        "city":              "Delhi",
        "forecast_date":     pd.date_range("2023-03-02", periods=10),
        "temp_mean_pred":    [28.0 + i * 0.5 for i in range(10)],
        "temp_max_pred":     [32.0 + i * 0.5 for i in range(10)],
        "temp_min_pred":     [22.0 + i * 0.3 for i in range(10)],
        "precipitation_pred":[2.0] * 10,
        "confidence_lower":  [25.0] * 10,
        "confidence_upper":  [31.0] * 10,
    })


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestClimateData:
    def test_upsert_and_retrieve(self, sample_climate_df):
        rows = upsert_climate_data(sample_climate_df)
        assert rows == 60

        df = get_climate_data("Delhi")
        assert len(df) == 60
        assert "temp_mean" in df.columns

    def test_date_filter(self, sample_climate_df):
        upsert_climate_data(sample_climate_df)
        df = get_climate_data("Delhi", start_date="2023-02-01", end_date="2023-02-28")
        assert len(df) == 28

    def test_upsert_idempotent(self, sample_climate_df):
        upsert_climate_data(sample_climate_df)
        upsert_climate_data(sample_climate_df)  # same data again
        df = get_climate_data("Delhi")
        assert len(df) == 60  # no duplicates

    def test_missing_required_columns_raises(self):
        bad_df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3)})
        with pytest.raises(ValueError):
            upsert_climate_data(bad_df)

    def test_unknown_city_returns_empty(self, sample_climate_df):
        upsert_climate_data(sample_climate_df)
        df = get_climate_data("Atlantis")
        assert df.empty


class TestAnomalyScores:
    def test_upsert_and_retrieve(self, sample_climate_df, sample_anomaly_df):
        upsert_climate_data(sample_climate_df)
        rows = upsert_anomaly_scores(sample_anomaly_df)
        assert rows == 30

        df = get_anomaly_scores("Delhi")
        assert len(df) == 30
        assert df["is_anomaly"].sum() == 2

    def test_anomaly_type_column(self, sample_climate_df, sample_anomaly_df):
        upsert_climate_data(sample_climate_df)
        upsert_anomaly_scores(sample_anomaly_df)
        df = get_anomaly_scores("Delhi")
        assert "heatwave" in df["anomaly_type"].values


class TestForecasts:
    def test_upsert_and_latest(self, sample_climate_df, sample_forecast_df):
        upsert_climate_data(sample_climate_df)
        rows = upsert_forecasts(sample_forecast_df)
        assert rows == 10

        df = get_latest_forecast("Delhi")
        assert len(df) == 10
        assert "temp_mean_pred" in df.columns


class TestDBStats:
    def test_stats_keys(self, sample_climate_df):
        upsert_climate_data(sample_climate_df)
        stats = get_db_stats()
        assert "climate_rows" in stats
        assert "cities" in stats
        assert "Delhi" in stats["cities"]

"""
tests/test_anomaly_detector.py
────────────────────────────────
Unit tests for the Spatiotemporal Autoencoder anomaly detector.
Uses mock data (no GEE calls) so tests run offline.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override model save directory to a temp dir
TEST_MODEL_DIR = tempfile.mkdtemp()
os.environ["MODEL_SAVE_DIR_OVERRIDE"] = TEST_MODEL_DIR

from config import SEQ_LENGTH
from models.anomaly_detector import (
    SpatiotemporalAnomalyDetector,
    _classify_anomaly_type,
    _score_to_severity,
    _make_sequences,
)


@pytest.fixture
def mock_climate_df():
    """Minimal mock climate dataframe for testing."""
    np.random.seed(42)
    n = SEQ_LENGTH * 6   # at least 6× the sequence length
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "date":            dates,
        "city":            "Delhi",
        "temp_max":        30 + np.random.normal(0, 3, n),
        "temp_min":        20 + np.random.normal(0, 2, n),
        "temp_mean":       25 + np.random.normal(0, 2.5, n),
        "precipitation":   np.maximum(0, np.random.exponential(3, n)),
        "wind_speed":      np.abs(np.random.normal(3, 1, n)),
        "solar_radiation": 15e6 + np.random.normal(0, 5e5, n),
    })


class TestMakeSequences:
    def test_output_shape(self):
        data = np.random.rand(100, 6)
        X = _make_sequences(data, 30)
        assert X.shape == (71, 30, 6)  # 100 - 30 + 1 = 71

    def test_minimum_length(self):
        data = np.random.rand(30, 6)
        X = _make_sequences(data, 30)
        assert X.shape[0] == 1


class TestClassifyAnomalyType:
    def test_heatwave(self):
        row = {"is_anomaly": True, "temp_mean": 42, "precipitation": 0.5}
        assert _classify_anomaly_type(row) == "heatwave"

    def test_heavy_rain(self):
        row = {"is_anomaly": True, "temp_mean": 30, "precipitation": 80}
        assert _classify_anomaly_type(row) == "heavy_rain"

    def test_cold_spell(self):
        row = {"is_anomaly": True, "temp_mean": 8, "precipitation": 2}
        assert _classify_anomaly_type(row) == "cold_spell"

    def test_none_when_not_anomaly(self):
        row = {"is_anomaly": False, "temp_mean": 42, "precipitation": 0}
        assert _classify_anomaly_type(row) == "none"


class TestScoreToSeverity:
    def test_thresholds(self):
        assert _score_to_severity(0.1)  == "low"
        assert _score_to_severity(0.4)  == "medium"
        assert _score_to_severity(0.65) == "high"
        assert _score_to_severity(0.95) == "extreme"


class TestSpatiotemporalAnomalyDetector:
    def test_init(self):
        det = SpatiotemporalAnomalyDetector("Delhi")
        assert det.city == "Delhi"
        assert det.model is None
        assert det.threshold is None

    def test_train_and_score(self, mock_climate_df):
        det = SpatiotemporalAnomalyDetector("Delhi")
        det.model_path = os.path.join(TEST_MODEL_DIR, "stae_test_Delhi.keras")
        det.meta_path  = os.path.join(TEST_MODEL_DIR, "stae_test_Delhi_meta.pkl")

        # Use fewer epochs for test speed
        import config as cfg
        original_epochs = cfg.EPOCHS
        cfg.EPOCHS = 2

        det.train(mock_climate_df)

        cfg.EPOCHS = original_epochs

        assert det.model is not None
        assert det.threshold is not None
        assert det.threshold > 0

        scores = det.score(mock_climate_df)
        assert "anomaly_score" in scores.columns
        assert "is_anomaly" in scores.columns
        assert "severity" in scores.columns
        assert "anomaly_type" in scores.columns
        assert len(scores) > 0
        assert scores["anomaly_score"].between(0, 1).all()

    def test_load_fails_gracefully(self):
        det = SpatiotemporalAnomalyDetector("NonExistentCity")
        result = det.load()
        assert result is False

    def test_score_without_train_raises(self, mock_climate_df):
        det = SpatiotemporalAnomalyDetector("Mumbai")
        det.model_path = "/tmp/nonexistent_stae.keras"
        det.meta_path  = "/tmp/nonexistent_stae_meta.pkl"
        with pytest.raises(RuntimeError, match="No trained model"):
            det.score(mock_climate_df)

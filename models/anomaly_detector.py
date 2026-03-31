"""
models/anomaly_detector.py
──────────────────────────
Spatiotemporal Autoencoder (STAE) + Learned Anomaly Classifier
Production Grid Edition — 8-feature input.

Upgrades from v2:
  - 8-feature input (soil_moisture + lai_high_veg added)
  - Adaptive per-grid-point thresholds replacing global 65th percentile
  - RandomForestClassifier replaces hardcoded heuristic type classification
  - grid_id-keyed model persistence
"""
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SEQ_LENGTH, ANOMALY_THRESHOLD, BATCH_SIZE, EPOCHS,
    MODEL_SAVE_DIR, FEATURE_COLS, N_FEATURES,
)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def _build_autoencoder(seq_len: int, n_features: int):
    """8-feature Bidirectional LSTM Autoencoder."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(seq_len, n_features))

    # ── Encoder ──────────────────────────────────────────
    x       = layers.Dense(32, activation="relu")(inputs)
    x       = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    encoded = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)

    # ── Bottleneck ───────────────────────────────────────
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.RepeatVector(seq_len)(x)

    # ── Decoder ──────────────────────────────────────────
    x       = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x       = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    decoded = layers.TimeDistributed(layers.Dense(n_features))(x)

    autoencoder = keras.Model(inputs, decoded, name="STAE_8feat")
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return autoencoder


def _build_encoder_only(full_model):
    """Extract encoder half of the STAE for bottleneck feature extraction."""
    from tensorflow import keras
    # Encoder outputs are at the RepeatVector input — layer index 5
    encoder = keras.Model(
        inputs=full_model.input,
        outputs=full_model.layers[5].input,  # Dense before RepeatVector
        name="STAE_Encoder",
    )
    return encoder


def _make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([data[i: i + seq_len] for i in range(len(data) - seq_len + 1)])


class SpatiotemporalAnomalyDetector:
    """
    8-feature STAE with adaptive threshold + learned RF anomaly classifier.
    Grid-id keyed — replaces city-keyed v2 version.
    """

    def __init__(self, grid_id: str):
        self.grid_id     = grid_id
        self.scaler      = None
        self.model       = None
        self.encoder     = None
        self.threshold   = None
        self.classifier  = None           # Learned anomaly type classifier
        self.model_path  = os.path.join(MODEL_SAVE_DIR, f"stae_{grid_id}.keras")
        self.meta_path   = os.path.join(MODEL_SAVE_DIR, f"stae_{grid_id}_meta.pkl")

    def _fit_scale(self, df: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        cols = [c for c in FEATURE_COLS if c in df.columns]
        return self.scaler.fit_transform(df[cols].fillna(df[cols].mean()))

    def _apply_scale(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        return self.scaler.transform(df[cols].fillna(df[cols].mean()))

    def _adaptive_threshold(self, errors: np.ndarray, df: pd.DataFrame) -> float:
        """
        Compute an adaptive anomaly threshold.
        Adjusts the base percentile upward for high-variance grid cells
        (e.g., desert/coastal regions have naturally more climate variability).
        """
        # Measure natural variability via temperature range std
        temp_range_std = df["temp_max"].std() - df["temp_min"].std() if "temp_max" in df.columns else 0
        # Bump the base percentile up by up to 10 points for high-variance cells
        variance_adj = min(10, abs(temp_range_std) * 0.5)
        adjusted_pct = ANOMALY_THRESHOLD * 100 + variance_adj
        threshold    = float(np.percentile(errors, adjusted_pct))
        logger.info(f"[{self.grid_id}] Adaptive threshold = {threshold:.6f} "
                    f"(pct={adjusted_pct:.1f}, variance_adj={variance_adj:.2f})")
        return threshold

    def _train_classifier(self, X: np.ndarray, df: pd.DataFrame):
        """
        Train a lightweight RandomForest classifier on STAE bottleneck features
        to predict anomaly type, replacing the old hardcoded heuristic.
        """
        from sklearn.ensemble import RandomForestClassifier

        # Generate soft labels using the heuristic — used as training signal
        labels = df.apply(_heuristic_label, axis=1).values[SEQ_LENGTH - 1:]

        try:
            bottleneck = self.encoder.predict(X, verbose=0)
            valid = labels != "none"
            if valid.sum() < 10:
                logger.warning(f"[{self.grid_id}] Too few anomaly samples for RF. Using heuristic fallback.")
                self.classifier = None
                return
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.classifier.fit(bottleneck, labels)
            logger.info(f"[{self.grid_id}] RF classifier trained on {len(labels)} samples.")
        except Exception as e:
            logger.warning(f"[{self.grid_id}] RF classifier training failed: {e}")
            self.classifier = None

    def train(self, df: pd.DataFrame) -> "SpatiotemporalAnomalyDetector":
        import tensorflow as tf

        logger.info(f"[{self.grid_id}] Training 8-feat STAE on {len(df)} records ...")
        data_scaled = self._fit_scale(df)
        X           = _make_sequences(data_scaled, SEQ_LENGTH)
        n_feat      = data_scaled.shape[1]

        self.model = _build_autoencoder(SEQ_LENGTH, n_feat)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(
            X, X,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, callbacks=[early_stop], verbose=0,
        )

        # Build encoder for classifier
        try:
            self.encoder = _build_encoder_only(self.model)
        except Exception:
            self.encoder = None

        recon  = self.model.predict(X, verbose=0)
        errors = np.mean(np.abs(X - recon), axis=(1, 2))

        # Adaptive threshold
        aligned_df  = df.iloc[SEQ_LENGTH - 1:].reset_index(drop=True)
        self.threshold = self._adaptive_threshold(errors, aligned_df)

        # Learned classifier
        if self.encoder is not None:
            self._train_classifier(X, aligned_df)

        self._save()
        return self

    def _save(self):
        self.model.save(self.model_path)
        joblib.dump({
            "scaler":     self.scaler,
            "threshold":  self.threshold,
            "classifier": self.classifier,
            "encoder":    None,   # encoder rebuilt from model on load
        }, self.meta_path)
        logger.info(f"[{self.grid_id}] STAE saved → {self.model_path}")

    def load(self) -> bool:
        if not (os.path.exists(self.model_path) and os.path.exists(self.meta_path)):
            return False
        from tensorflow.keras.models import load_model
        self.model = load_model(self.model_path)
        meta = joblib.load(self.meta_path)
        self.scaler     = meta["scaler"]
        self.threshold  = meta["threshold"]
        self.classifier = meta.get("classifier")
        try:
            self.encoder = _build_encoder_only(self.model)
        except Exception:
            self.encoder = None
        logger.info(f"[{self.grid_id}] STAE loaded from disk.")
        return True

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each day in df.
        Returns [date, grid_id, recon_error, anomaly_score, is_anomaly,
                 anomaly_type, severity]
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError(f"No trained model for {self.grid_id}. Run .train() first.")

        data_scaled = self._apply_scale(df)
        X           = _make_sequences(data_scaled, SEQ_LENGTH)
        recon       = self.model.predict(X, verbose=0)
        errors      = np.mean(np.abs(X - recon), axis=(1, 2))

        aligned_df = df.iloc[SEQ_LENGTH - 1:].copy().reset_index(drop=True)
        aligned_df = aligned_df.iloc[:len(errors)]

        anomaly_scores = np.clip(errors / (self.threshold * 2 + 1e-8), 0, 1)

        aligned_df["recon_error"]   = errors
        aligned_df["anomaly_score"] = anomaly_scores
        aligned_df["is_anomaly"]    = errors > self.threshold

        # Anomaly type classification
        if self.classifier is not None and self.encoder is not None:
            try:
                bottleneck = self.encoder.predict(X, verbose=0)
                aligned_df["anomaly_type"] = self.classifier.predict(bottleneck)
            except Exception:
                aligned_df["anomaly_type"] = aligned_df.apply(_heuristic_label, axis=1)
        else:
            aligned_df["anomaly_type"] = aligned_df.apply(_heuristic_label, axis=1)

        aligned_df["severity"] = anomaly_scores.tolist()
        aligned_df["severity"] = aligned_df["anomaly_score"].apply(_score_to_severity)

        id_col = "grid_id" if "grid_id" in aligned_df.columns else "city"
        return_cols = ["date", id_col, "recon_error", "anomaly_score",
                       "is_anomaly", "anomaly_type", "severity"]
        return_cols = [c for c in return_cols if c in aligned_df.columns]
        return aligned_df[return_cols]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _heuristic_label(row) -> str:
    """Soft label for classifier training — also used as fallback."""
    temp    = row.get("temp_mean", 25)
    precip  = row.get("precipitation", 0)
    sm      = row.get("soil_moisture", 0.3)
    lai     = row.get("lai_high_veg", 2.0)

    if temp > 38 and precip < 2 and sm < 0.15:
        return "heatwave"
    if precip > 40:
        return "heavy_rain"
    if temp < 8:
        return "cold_spell"
    if sm < 0.1 and lai < 1.0:
        return "drought"
    return "compound"


def _score_to_severity(score: float) -> str:
    if score < 0.3:  return "low"
    if score < 0.55: return "medium"
    if score < 0.80: return "high"
    return "extreme"

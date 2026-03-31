"""
models/forecaster.py
────────────────────
Ensemble BiLSTM + Temporal Attention Forecaster — Production Grid Edition.

Upgrades from v2:
  - Temporal Attention layer between BiLSTM encoder stages
  - 3-model ensemble with different seeds for variance reduction
  - 8-feature input (added soil_moisture, lai_high_veg)
  - Monte Carlo Dropout for uncertainty intervals
"""
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SEQ_LENGTH, FORECAST_HORIZON, BATCH_SIZE, EPOCHS,
    MODEL_SAVE_DIR, FEATURE_COLS, N_FEATURES, ENSEMBLE_SIZE,
)

OUTPUT_COLS = ["temp_mean", "temp_max", "temp_min", "precipitation"]
N_OUT       = len(OUTPUT_COLS)
MC_SAMPLES  = 30

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def _build_attention_bilstm(seq_len: int, n_in: int, n_out: int,
                             horizon: int, seed: int = 42):
    """BiLSTM + Temporal Self-Attention forecaster (single ensemble member)."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Input, Model

    inputs = Input(shape=(seq_len, n_in))

    # Encoder stage 1
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x, training=True)   # MC Dropout — active at inference

    # ── Temporal Self-Attention (key upgrade from prototype) ─────────────────
    attention_out = layers.Attention()([x, x])
    x = layers.Add()([x, attention_out])         # Residual connection
    x = layers.LayerNormalization()(x)
    # ────────────────────────────────────────────────────────────────────────

    # Encoder stage 2
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x, training=True)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    outputs = layers.Dense(horizon * n_out)(x)
    outputs = layers.Reshape((horizon, n_out))(outputs)

    model = Model(inputs, outputs, name=f"Attention_BiLSTM_seed{seed}")
    model.compile(
        optimizer="adam",
        loss="huber",
        metrics=["mae"],
    )
    return model


def _make_xy(data: np.ndarray, out_data: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i: i + seq_len])
        y.append(out_data[i + seq_len: i + seq_len + horizon])
    return np.array(X), np.array(y)


class ClimateForecaster:
    """
    3-model ensemble Attention-BiLSTM forecaster with MC Dropout intervals.
    Drop-in replacement for the original ClimateForecaster.
    """

    def __init__(self, grid_id: str):
        self.grid_id     = grid_id
        self.models: List = []
        self.in_scaler   = None
        self.out_scaler  = None
        self.model_dir   = MODEL_SAVE_DIR

    def _paths(self, member: int):
        base = os.path.join(self.model_dir, f"forecaster_{self.grid_id}_m{member}")
        return f"{base}.keras", f"{base}_meta.pkl"

    def train(self, df: pd.DataFrame) -> "ClimateForecaster":
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf

        logger.info(f"[{self.grid_id}] Training {ENSEMBLE_SIZE}-model ensemble "
                    f"Attention-BiLSTM on {len(df)} records ...")

        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        tgt_cols  = [c for c in OUTPUT_COLS   if c in df.columns]

        in_data  = df[feat_cols].fillna(df[feat_cols].mean()).values
        out_data = df[tgt_cols].fillna(df[tgt_cols].mean()).values

        self.in_scaler  = StandardScaler()
        self.out_scaler = StandardScaler()
        in_scaled  = self.in_scaler.fit_transform(in_data)
        out_scaled = self.out_scaler.fit_transform(out_data)

        X, y = _make_xy(in_scaled, out_scaled, SEQ_LENGTH, FORECAST_HORIZON)

        self.models = []
        seeds = [42, 123, 2024][:ENSEMBLE_SIZE]
        for i, seed in enumerate(seeds):
            logger.info(f"[{self.grid_id}] Training ensemble member {i+1}/{ENSEMBLE_SIZE} ...")
            model = _build_attention_bilstm(
                SEQ_LENGTH, in_scaled.shape[1], len(tgt_cols), FORECAST_HORIZON, seed
            )
            cbs = [
                tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            ]
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1, callbacks=cbs, verbose=0)
            self.models.append(model)

        self._save()
        return self

    def _save(self):
        meta = {"in_scaler": self.in_scaler, "out_scaler": self.out_scaler}
        for i, model in enumerate(self.models):
            m_path, meta_path = self._paths(i)
            model.save(m_path)
            joblib.dump(meta, meta_path)
        logger.info(f"[{self.grid_id}] Ensemble saved ({len(self.models)} members).")

    def load(self) -> bool:
        from tensorflow.keras.models import load_model
        self.models = []
        for i in range(ENSEMBLE_SIZE):
            m_path, meta_path = self._paths(i)
            if not (os.path.exists(m_path) and os.path.exists(meta_path)):
                break
            self.models.append(load_model(m_path))
            meta = joblib.load(meta_path)
        if self.models:
            self.in_scaler  = meta["in_scaler"]
            self.out_scaler = meta["out_scaler"]
            logger.info(f"[{self.grid_id}] Loaded {len(self.models)}-model ensemble.")
            return True
        return False

    def predict(self, df: pd.DataFrame, start_date: str = None) -> pd.DataFrame:
        """
        Predict next FORECAST_HORIZON days.
        Uses ensemble mean + MC Dropout for combined uncertainty.
        """
        if not self.models:
            if not self.load():
                raise RuntimeError(f"No trained model for {self.grid_id}. Run .train() first.")

        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        in_data   = df[feat_cols].fillna(df[feat_cols].mean()).values
        in_scaled = self.in_scaler.transform(in_data)
        window    = in_scaled[-SEQ_LENGTH:][np.newaxis, ...]   # (1, seq_len, n_in)

        # Ensemble × MC Dropout: shape (ENSEMBLE_SIZE × MC_SAMPLES, horizon, n_out)
        all_preds = []
        for model in self.models:
            mc = np.array([model.predict(window, verbose=0)[0] for _ in range(MC_SAMPLES)])
            all_preds.append(mc)
        all_preds = np.concatenate(all_preds, axis=0)  # (total_samples, horizon, n_out)

        mean_pred = all_preds.mean(axis=0)
        std_pred  = all_preds.std(axis=0)

        mean_inv = self.out_scaler.inverse_transform(mean_pred)
        lo_inv   = self.out_scaler.inverse_transform(mean_pred - 1.96 * std_pred)
        hi_inv   = self.out_scaler.inverse_transform(mean_pred + 1.96 * std_pred)

        last_date      = pd.to_datetime(start_date or df["date"].max())
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON)

        tgt_cols = [c for c in OUTPUT_COLS if c in df.columns]
        result   = pd.DataFrame({"forecast_date": forecast_dates})
        for i, col in enumerate(tgt_cols):
            result[f"{col}_pred"]     = np.round(mean_inv[:, i], 2)
            result[f"{col}_lower_95"] = np.round(lo_inv[:, i], 2)
            result[f"{col}_upper_95"] = np.round(hi_inv[:, i], 2)

        result.insert(0, "grid_id",      self.grid_id)
        result.insert(1, "generated_at", pd.Timestamp.utcnow())

        # Add aliases expected by dashboard
        if "temp_mean_pred" in result.columns:
            result["temp_mean_lower_95"] = result["temp_mean_lower_95"]
            result["temp_mean_upper_95"] = result["temp_mean_upper_95"]

        return result

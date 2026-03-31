"""
models/shap_explainer.py
────────────────────────
Async-safe SHAP KernelExplainer module for the production grid system.

Provides:
  - compute_shap_importance(): returns per-feature importance dict
  - generate_anomaly_briefing(): synthesizes natural-language summary
  - run_shap_for_grid(): full pipeline callable from API background tasks
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_COLS, SEQ_LENGTH, SHAP_BG_SAMPLES, SHAP_EVAL_SAMPLES

ANOMALY_TYPE_EMOJI = {
    "heatwave":   "🔥",
    "heavy_rain": "🌧️",
    "cold_spell": "❄️",
    "drought":    "🌵",
    "compound":   "⚡",
    "none":       "✅",
}


def _flat_predict(stae_model, scaler, X_flat: np.ndarray) -> np.ndarray:
    """
    Black-box prediction wrapper for SHAP KernelExplainer.
    Input:  2D array (n_samples, seq_len * n_features)
    Output: 1D array (n_samples,) of reconstruction errors
    """
    n_features = len(FEATURE_COLS)
    X_3d = X_flat.reshape(-1, SEQ_LENGTH, n_features)
    recon = stae_model.predict(X_3d, verbose=0)
    errors = np.mean(np.abs(X_3d - recon), axis=(1, 2))
    return errors


def compute_shap_importance(
    stae_model,
    scaler,
    background_data: np.ndarray,
    anomalous_data: np.ndarray,
) -> dict:
    """
    Compute SHAP feature importances for anomalous time-step(s).

    Args:
        stae_model:      Trained Keras STAE model
        scaler:          Fitted sklearn scaler used during training
        background_data: Shape (n_bg, SEQ_LENGTH, N_FEATURES) — normal samples
        anomalous_data:  Shape (n_eval, SEQ_LENGTH, N_FEATURES) — flagged samples

    Returns:
        dict: {"feature_name": mean_abs_shap_value, ...} sorted descending
    """
    try:
        import shap

        n_feat = len(FEATURE_COLS)
        # Flatten 3D → 2D for KernelExplainer
        bg_flat   = background_data.reshape(len(background_data), -1)
        eval_flat = anomalous_data.reshape(len(anomalous_data), -1)

        predict_fn = lambda X: _flat_predict(stae_model, scaler, X)

        # Use KMeans summary for efficiency (avoids huge background sets)
        bg_summary = shap.kmeans(bg_flat, min(10, len(bg_flat)))

        explainer   = shap.KernelExplainer(predict_fn, bg_summary)
        shap_values = explainer.shap_values(eval_flat, nsamples=100)

        # shap_values shape: (n_eval, seq_len * n_features)
        # Reshape and average over the time dimension
        sv_3d = np.array(shap_values).reshape(-1, SEQ_LENGTH, n_feat)
        mean_abs = np.mean(np.abs(sv_3d), axis=(0, 1))   # (n_features,)

        # Normalize to percentages
        total = mean_abs.sum() + 1e-10
        importance = {
            FEATURE_COLS[i]: float(round(mean_abs[i] / total, 4))
            for i in range(n_feat)
        }
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
        logger.info(f"SHAP importance computed: {importance}")
        return importance

    except ImportError:
        logger.warning("shap not installed — returning uniform importance.")
        return {f: round(1.0 / len(FEATURE_COLS), 4) for f in FEATURE_COLS}
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {}


def generate_anomaly_briefing(
    grid_id: str,
    anomaly_type: str,
    severity: str,
    anomaly_score: float,
    shap_importance: dict,
    date_str: str = "",
) -> str:
    """
    Synthesize a natural-language anomaly intelligence briefing.
    Uses SHAP importances to name the top-2 driving variables.
    """
    emoji   = ANOMALY_TYPE_EMOJI.get(anomaly_type, "⚡")
    sev_str = severity.upper()
    score_pct = int(anomaly_score * 100)

    # Top-2 SHAP drivers
    top_drivers = list(shap_importance.items())[:2] if shap_importance else []
    driver_text = ""
    if top_drivers:
        parts = []
        for feat, importance in top_drivers:
            feat_label = feat.replace("_", " ").title()
            parts.append(f"{feat_label} ({int(importance * 100)}%)")
        driver_text = f" Primary drivers: {' and '.join(parts)}."

    if anomaly_type == "none" or severity == "low":
        return f"✅ {grid_id}: Climate conditions are within normal bounds."

    return (
        f"{emoji} [{sev_str}] {grid_id}{' — ' + date_str if date_str else ''}: "
        f"A {anomaly_type.replace('_', ' ')} event detected with "
        f"{score_pct}% anomaly confidence.{driver_text}"
    )


def run_shap_for_grid(
    grid_id: str,
    stae_model,
    scaler,
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> dict:
    """
    Full SHAP pipeline for a grid point. Called asynchronously by the API.

    Returns dict mapping date → {"shap_importance": {...}, "briefing": "..."}
    """
    from config import FEATURE_COLS

    results = {}
    data_scaled = scaler.transform(
        df[[c for c in FEATURE_COLS if c in df.columns]].fillna(0).values
    )

    # Build background from non-anomalous days
    normal_days = anomaly_df[~anomaly_df["is_anomaly"]].index.tolist()
    if len(normal_days) < 5:
        logger.warning(f"[{grid_id}] Insufficient normal days for SHAP background.")
        return results

    bg_indices = normal_days[:SHAP_BG_SAMPLES]
    bg_seqs    = np.array([
        data_scaled[i: i + SEQ_LENGTH]
        for i in bg_indices
        if i + SEQ_LENGTH <= len(data_scaled)
    ])
    if len(bg_seqs) < 2:
        return results

    # Evaluate on first N anomalous days
    anom_indices = anomaly_df[anomaly_df["is_anomaly"]].index.tolist()[:SHAP_EVAL_SAMPLES]
    for idx in anom_indices:
        if idx + SEQ_LENGTH > len(data_scaled):
            continue
        eval_seq   = data_scaled[idx: idx + SEQ_LENGTH][np.newaxis, ...]
        row        = anomaly_df.iloc[idx]
        importance = compute_shap_importance(stae_model, scaler, bg_seqs, eval_seq)
        briefing   = generate_anomaly_briefing(
            grid_id=grid_id,
            anomaly_type=str(row.get("anomaly_type", "compound")),
            severity=str(row.get("severity", "medium")),
            anomaly_score=float(row.get("anomaly_score", 0.5)),
            shap_importance=importance,
            date_str=str(row.get("date", "")),
        )
        date_key = str(row.get("date", idx))
        results[date_key] = {"shap_importance": importance, "briefing": briefing}

    return results

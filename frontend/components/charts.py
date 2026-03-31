"""
frontend/components/charts.py
──────────────────────────────
Plotly chart components for the Climate AI dashboard.

Upgrades:
  - Replaced pie chart with horizontal severity bar chart
  - Upgraded precipitation to area chart with cumulative rainfall
  - Added SHAP importance horizontal bar chart
  - Interactive themes
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

THEME = dict(
    bg="rgba(0,0,0,0)",       # transparent for glassmorphism
    plot_bg="rgba(0,0,0,0)",
    grid="rgba(255,255,255,0.05)",
    text="#E6EDF3",
    accent="#00C6FF",
    anomaly_red="#FF4444",
    safe_green="#39D353",
    warn_yellow="#FFA500",
)


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=THEME["text"], size=15, family="Inter")),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text"], family="Inter"),
        xaxis=dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )


def render_temperature_chart(df: pd.DataFrame, anomaly_df: pd.DataFrame = None):
    """Render a temperature trend chart with anomaly markers."""
    fig = go.Figure()

    if "temp_max" in df.columns and "temp_min" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["temp_max"],
            mode="lines", line=dict(color="rgba(0,198,255,0)", width=0),
            name="Temp Max", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["temp_min"],
            fill="tonexty", mode="lines",
            line=dict(color="rgba(0,198,255,0)", width=0),
            fillcolor="rgba(0,198,255,0.08)", name="Temp Range",
        ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["temp_mean"],
        mode="lines", name="Temp Mean",
        line=dict(color=THEME["accent"], width=2.5),
    ))

    if anomaly_df is not None and not anomaly_df.empty:
        anom = anomaly_df[anomaly_df["is_anomaly"] == True]
        if not anom.empty:
            merged = anom.merge(df[["date", "temp_mean"]], on="date", how="left")
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["temp_mean"],
                mode="markers", name="⚠️ Anomaly",
                marker=dict(color=THEME["anomaly_red"], size=12, symbol="circle-dot", line=dict(width=2, color="white")),
            ))

    fig.update_layout(**_base_layout("🌡️ Temperature Trend"))
    st.plotly_chart(fig, use_container_width=True)


def render_precipitation_chart(df: pd.DataFrame, anomaly_df: pd.DataFrame = None):
    """Render precipitation area chart with cumulative overlay."""
    fig = go.Figure()

    # Daily Area
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["precipitation"],
        fill='tozeroy', fillcolor='rgba(0,198,255,0.15)',
        mode="lines",
        line=dict(color='#00C6FF', width=1.5),
        name='Daily Rain',
    ))

    # Cumulative on secondary axis
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["precipitation"].cumsum(),
        mode="lines",
        line=dict(color='#FF6B6B', width=2, dash='dot'),
        name='Cumulative (mm)',
        yaxis='y2'
    ))

    if anomaly_df is not None and not anomaly_df.empty:
        heavy = anomaly_df[anomaly_df["anomaly_type"] == "heavy_rain"]
        if not heavy.empty:
            for _, row in heavy.iterrows():
                fig.add_vrect(
                    x0=row["date"], x1=row["date"],
                    fillcolor="rgba(255,68,68,0.3)", layer="below", line_width=0,
                )

    layout = _base_layout("🌧️ Precipitation Analysis")
    layout["yaxis2"] = dict(
        title=dict(text="Cumulative (mm)", font=dict(color="#FF6B6B")), 
        overlaying="y", side="right",
        gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#FF6B6B")
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_score_chart(anomaly_df: pd.DataFrame):
    """Render the anomaly score timeline."""
    if anomaly_df is None or anomaly_df.empty:
        st.info("No anomaly scores available yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomaly_df["date"], y=anomaly_df["anomaly_score"],
        mode="lines+markers", name="Anomaly Score",
        line=dict(color=THEME["warn_yellow"], width=2),
        marker=dict(
            size=6,
            color=anomaly_df["is_anomaly"].map({True: THEME["anomaly_red"], False: THEME["safe_green"]}).fillna(THEME["safe_green"]),
        ),
    ))

    fig.add_hline(y=0.5, line_dash="dot", line_color=THEME["anomaly_red"],
                  annotation_text="Threshold", annotation_position="top left")

    fig.update_layout(**_base_layout("📊 Anomaly Score Timeline"), yaxis_range=[0, min(1.2, anomaly_df["anomaly_score"].max() + 0.2)])
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_chart(forecast_df: pd.DataFrame, historical_df: pd.DataFrame = None):
    """Render 10-day ensemble forecast."""
    fig = go.Figure()

    if historical_df is not None and not historical_df.empty:
        recent = historical_df.tail(30)
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["temp_mean"],
            mode="lines", name="Historical",
            line=dict(color=THEME["accent"], width=2),
        ))

    pred_col  = next((c for c in forecast_df.columns if "temp_mean_pred" in c), None)
    lower_col = next((c for c in forecast_df.columns if "lower" in c), None)
    upper_col = next((c for c in forecast_df.columns if "upper" in c), None)

    if pred_col:
        fig.add_trace(go.Scatter(
            x=forecast_df["forecast_date"], y=forecast_df[pred_col],
            mode="lines+markers", name="Forecast",
            line=dict(color="#FF6B6B", width=2.5, dash="dash"),
            marker=dict(size=7),
        ))

    if lower_col and upper_col:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["forecast_date"], forecast_df["forecast_date"][::-1]]),
            y=pd.concat([forecast_df[upper_col], forecast_df[lower_col][::-1]]),
            fill="toself", fillcolor="rgba(255,107,107,0.15)",
            line=dict(color="rgba(255,107,107,0)"),
            name="95% Confidence",
        ))

    fig.update_layout(**_base_layout("🔮 10-Day Ensemble Forecast"))
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_type_breakdown(anomaly_df: pd.DataFrame):
    """Horizontal bar chart for anomaly types (replaces pie chart)."""
    if anomaly_df is None or anomaly_df.empty:
        return
    anom = anomaly_df[anomaly_df["is_anomaly"] == True]
    if anom.empty:
        st.success("✅ No anomalies detected in the selected period.")
        return

    counts = anom["anomaly_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    counts["type_label"] = counts["type"].str.replace("_", " ").str.title()
    counts = counts.sort_values("count", ascending=True)  # Plotly horizontal draws bottom-up

    fig = go.Figure(go.Bar(
        y=counts["type_label"],
        x=counts["count"],
        orientation='h',
        marker=dict(
            color=counts["count"],
            colorscale=[[0, THEME["warn_yellow"]], [1, THEME["anomaly_red"]]],
        ),
        text=counts["count"],
        textposition='outside',
        textfont=dict(color="white")
    ))

    fig.update_layout(**_base_layout("⚡ Anomaly Distribution"))
    fig.update_xaxes(title="Days")
    st.plotly_chart(fig, use_container_width=True)


def render_shap_waterfall(shap_importance: dict):
    """Horizontal bar representing SHAP feature attribution."""
    if not shap_importance:
        st.info("No SHAP values available for this event.")
        return

    df = pd.DataFrame(list(shap_importance.items()), columns=["Feature", "Importance"])
    df["Feature"] = df["Feature"].str.replace("_", " ").str.title()
    df = df.sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"] * 100,
        y=df["Feature"],
        orientation='h',
        marker=dict(color="#9D4EDD"),
        text=(df["Importance"] * 100).round(1).astype(str) + "%",
        textposition='auto',
    ))

    layout = _base_layout("🔬 SHAP Neural Driver Analysis")
    layout["xaxis"]["title"] = "Contribution %"
    layout["margin"] = dict(l=120, r=20, t=40, b=40)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame):
    """Plotly Correlation Heatmap for features."""
    if df.empty:
        return
    from config import FEATURE_COLS
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not cols:
        return

    corr = df[cols].corr()
    labels = [c.replace("_", " ").title() for c in cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        showscale=False,
    ))
    layout = _base_layout("🧬 Multi-Variable Correlation")
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

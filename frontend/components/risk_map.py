"""
frontend/components/risk_map.py
────────────────────────────────
Interactive PyDeck Heatmap / Scatter risk map.
Grid Edition: Adds HeatmapLayer for smooth gradient visualization.
"""
import pandas as pd
import pydeck as pdk
import streamlit as st

SEVERITY_COLORS = {
    "none":    [30, 200, 100, 160],
    "low":     [70, 180, 230, 180],
    "medium":  [255, 200, 0, 200],
    "high":    [255, 100, 0, 220],
    "extreme": [200, 0, 0, 240],
}

ANOMALY_TYPE_ICONS = {
    "heatwave":   "🔥",
    "heavy_rain": "🌧️",
    "cold_spell": "❄️",
    "drought":    "🌵",
    "compound":   "⚡",
    "none":       "✅",
}


def render_risk_map(overview_df: pd.DataFrame):
    """Render the grid risk overview map."""
    if overview_df.empty:
        st.warning("No overview data available yet. Run the data pipeline first.")
        return

    df = overview_df.copy()
    df["severity"] = df["severity"].fillna("none")
    df["color"]    = df["severity"].map(SEVERITY_COLORS).apply(
        lambda c: c if isinstance(c, list) else [30, 200, 100, 160]
    )
    # Radius based on resolution (~25km)
    df["radius"]   = 20000 
    df["tooltip_text"] = df.apply(
        lambda r: f"{r.get('grid_id', '')} | {r.get('severity', 'none').upper()} | "
                  f"{r.get('anomaly_type', 'none').replace('_',' ')}\n"
                  f"Temp: {r.get('latest_temp', '—'):.1f}°C | "
                  f"Precip: {r.get('latest_precip', '—'):.1f}mm | "
                  f"Soil: {r.get('soil_moisture', '—'):.2f}",
        axis=1,
    )

    # ── Scatter Layer (Clickable points) ─────────────────────────────────────────
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        opacity=0.8,
    )

    # ── Heatmap Layer (Thermal/Severity Gradients) ──────────────────────────────
    # We use anomaly_score for weight to show gradients of risk across central india
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df[df["anomaly_score"] > 0.05],
        opacity=0.6,
        get_position=["lon", "lat"],
        get_weight="anomaly_score * 100",
        radiusPixels=60,
        intensity=1.5,
        threshold=0.1,
        colorRange=[
            [30, 200, 100],  # Green
            [255, 200, 0],   # Yellow
            [255, 100, 0],   # Orange
            [200, 0, 0],     # Red
            [120, 0, 0]      # Dark Red
        ]
    )

    view = pdk.ViewState(
        latitude=df["lat"].mean() if not df.empty else 21.0,
        longitude=df["lon"].mean() if not df.empty else 79.0,
        zoom=4.8,
        pitch=45,
    )

    deck = pdk.Deck(
        layers=[heatmap_layer, scatter_layer],
        initial_view_state=view,
        tooltip={"text": "{tooltip_text}"},
        map_style="mapbox://styles/mapbox/dark-v11",
    )

    st.pydeck_chart(deck, use_container_width=True)

    # Legend
    st.markdown("""
    <div style='display:flex; justify-content:center; gap:20px; font-size:0.8rem; color:#8B949E; margin-top:-10px; margin-bottom:10px;'>
      <span><span style='color:rgb(30,200,100)'>●</span> Normal</span>
      <span><span style='color:rgb(70,180,230)'>●</span> Low</span>
      <span><span style='color:rgb(255,200,0)'>●</span> Medium</span>
      <span><span style='color:rgb(255,100,0)'>●</span> High</span>
      <span><span style='color:rgb(200,0,0)'>●</span> Extreme</span>
    </div>
    """, unsafe_allow_html=True)

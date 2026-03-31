"""
frontend/app.py
────────────────
Climate AI v2 — Simple AI Summary Dashboard
Focuses purely on the natural language intelligence briefings and AI narrative.
"""
import os
import sys
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_HOST, API_PORT
from frontend.components.charts import (
    render_anomaly_score_chart, 
    render_shap_waterfall, 
    render_temperature_chart, 
    render_precipitation_chart
)

API_BASE = f"http://{API_HOST}:{API_PORT}/api/v1"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate AI — Intelligence Summary",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Minimal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0D1117; color: #E6EDF3; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
  }

  /* Section headers */
  h1 {
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important; font-weight: 700 !important;
    text-align: center;
    margin-bottom: 2rem;
  }
  h2 { color: #E6EDF3 !important; font-size: 1.4rem !important; font-weight: 600 !important; }

  /* Briefing Box - Large and Readable */
  .briefing-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 32px;
    font-size: 1.1rem;
    line-height: 1.8;
    color: #C9D1D9;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }
  
  .insight-highlight {
    color: #58A6FF;
    font-weight: 600;
  }

  /* Alert pills */
  .alert-extreme { background: #3D1217; border-left: 4px solid #FF4444; border-radius:8px; padding:16px; margin-bottom: 16px; }
  .alert-high    { background: #2D1B0E; border-left: 4px solid #FFA500; border-radius:8px; padding:16px; margin-bottom: 16px; }
  .alert-medium  { background: #2B2208; border-left: 4px solid #FFD700; border-radius:8px; padding:16px; margin-bottom: 16px; }
  .alert-safe    { background: #0D2B1A; border-left: 4px solid #39D353; border-radius:8px; padding:16px; margin-bottom: 16px; }

  hr { border-color: #21262D !important; }
  
  .stButton > button {
    background: #238636;
    color: white; border: none; border-radius: 6px;
    font-weight: 600; width: 100%;
  }
  .stButton > button:hover { background: #2EA043; }
</style>
""", unsafe_allow_html=True)


# ─── API Fetch Helpers ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_unified_briefing() -> str:
    try:
        r = requests.get(f"{API_BASE}/anomaly/briefing", params={"days": 30}, timeout=8)
        return r.json().get("briefing", "") if r.status_code == 200 else "Briefing unavailable."
    except: return "Briefing unavailable."

@st.cache_data(ttl=300)
def fetch_anomaly_scores(grid_id: str, start: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/anomaly/scores/{grid_id}", params={"start_date": start}, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df["date"] = pd.to_datetime(df["date"])
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_grid_climate(grid_id: str, start: str, end: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/data/climate/{grid_id}", params={"start_date": start, "end_date": end}, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df["date"] = pd.to_datetime(df["date"])
            return df
    except: pass
    return pd.DataFrame()

def geocode_location(query: str) -> dict:
    try:
        r = requests.post(f"{API_BASE}/data/geocode", json={"query": query}, timeout=8)
        return r.json() if r.status_code == 200 else {}
    except: return {}

def check_api_status() -> bool:
    try: return requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=2).status_code == 200
    except: return False


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Climate AI")
    st.markdown("<small style='color:#8B949E'>Intelligence Summary View</small>", unsafe_allow_html=True)
    st.divider()

    api_ok = check_api_status()
    if api_ok: st.success("🟢 AI Engine Online")
    else:      st.error("🔴 AI Engine Offline — Demo mode")

    st.markdown("### 📍 Location Focus")
    
    if "selected_grid" not in st.session_state:
        st.session_state.selected_grid = "Grid_23.0_78.0"
    if "resolved_loc" not in st.session_state:
        st.session_state.resolved_loc = "Central India"

    search_query = st.text_input("Enter Pincode or City", placeholder="e.g. Pune")
    if st.button("Generate Local Report") and search_query:
        with st.spinner("Analyzing location..."):
            res = geocode_location(search_query)
            if res and "grid_id" in res:
                st.session_state.selected_grid = res["grid_id"]
                st.session_state.resolved_loc  = res["resolved"]
                st.success("Location synced.")
            else:
                st.error("Location outside grid coverage.")


# ─── Main Page ────────────────────────────────────────────────────────────────
st.title("🌍 Climate Intelligence Report")

st.markdown("### 🌐 National Overview (Last 30 Days)")
briefing = fetch_unified_briefing() if api_ok else "Demo mode: Multiple heatwave events detected in Central India driven by critically low soil moisture and consecutive 45°C+ days. Normal conditions observed in eastern sectors."

# Format briefing logically
html_briefing = briefing.replace('\n', '<br><br>').replace('Primary drivers:', '<span class="insight-highlight">Primary drivers:</span>')
st.markdown(f"<div class='briefing-box'>{html_briefing}</div>", unsafe_allow_html=True)

st.divider()

# ── Location Deep-Dive ────────────────────────────────────────────────────────
grid_id  = st.session_state.selected_grid
loc_name = st.session_state.resolved_loc

st.markdown(f"### 📍 Local Risk Analysis: **{loc_name}**")

start_str = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
end_str   = datetime.today().strftime("%Y-%m-%d")

anomaly_df  = fetch_anomaly_scores(grid_id, start_str)
climate_df  = fetch_grid_climate(grid_id, start_str, end_str)

# Demo Mode Fallback
if climate_df.empty:
    import numpy as np
    dates = pd.date_range(start_str, datetime.today())
    n = len(dates)
    climate_df = pd.DataFrame({
        "date": dates,
        "temp_mean": 30 + 5 * np.sin(np.linspace(0, 10, n)),
        "precipitation": np.random.exponential(5, n),
    })
    
if anomaly_df.empty:
    import numpy as np
    dates = pd.date_range(start_str, datetime.today())
    n = len(dates)
    anomaly_df = pd.DataFrame({
        "date": dates,
        "anomaly_score": np.random.uniform(0, 0.8, n),
        "is_anomaly": np.random.rand(n) < 0.1,
        "anomaly_type": "compound",
    })

# Active Alert Banner
if not anomaly_df.empty and "severity" in anomaly_df.columns:
    latest = anomaly_df.sort_values("date").iloc[-1]
    sev    = latest.get("severity", "none")
    atype  = latest.get("anomaly_type", "none").replace('_', ' ').title()
    
    if sev != "none":
        st.markdown(f"<div class='alert-{sev}'>🚨 **Active Threat Detected:** {atype} (Severity: {sev.upper()})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-safe'>✅ **Status:** Climate conditions are stable. No immediate threats detected.</div>", unsafe_allow_html=True)

# ── Simple Visual: 90-Day Anomaly Timeline
st.markdown("#### 📈 90-Day Historical Risk Timeline")
render_anomaly_score_chart(anomaly_df)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 🌡️ Physical Weather Context")
st.markdown("<p style='color:#8B949E; margin-top:-10px;'>A simple view of the physical temperatures contributing to the AI summary.</p>", unsafe_allow_html=True)
render_temperature_chart(climate_df, anomaly_df)

# SHAP Narrative Generation
active_anoms = anomaly_df[anomaly_df["is_anomaly"] == True]
if not active_anoms.empty and "shap_importance" in active_anoms.columns:
    latest_anom = active_anoms.iloc[-1]
    shap_data = latest_anom.get("shap_importance", {})
    
    if shap_data and len(shap_data) > 0:
        st.markdown("#### 🧠 AI Diagnostic Explanation")
        st.write(f"On **{latest_anom['date'].strftime('%B %d, %Y')}**, the deep learning engine flagged anomalous behavior. Here is the breakdown of what caused this classification:")
        
        # Turn SHAP JSON into a bulleted narrative list
        for feat, importance in sorted(shap_data.items(), key=lambda x: -x[1]):
            pct = int(importance * 100)
            if pct > 5:  # Only show meaningful drivers
                clean_name = feat.replace("_", " ").title()
                st.markdown(f"- **{pct}% Contribution:** Driven by irregular patterns in **{clean_name}**.")

        # ── Simple Visual: SHAP Waterfall
        st.markdown("<br>", unsafe_allow_html=True)
        render_shap_waterfall(shap_data)
else:
    if len(active_anoms) > 0:
        st.info("The AI engine recently flagged minor deviations, but no severe drivers were singled out for explanation.")


st.markdown("<br><br><br><center><small style='color:#8B949E'>Generated by Climate AI v2</small></center>", unsafe_allow_html=True)

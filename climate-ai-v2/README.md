# 🌍 Climate Intelligence Platform (v2)

An end-to-end, deep learning-powered spatiotemporal climate anomaly detection and early-warning system. 

Built to transition raw satellite environmental data into mathematically-derived, natural-language risk intelligence, empowering industries, governments, and local officials with proactive severe-weather foresight.

<div align="center">
  <h3>Powered by:</h3>
  <p><strong>FastAPI | Streamlit | DuckDB | TensorFlow | SHAP Keras | Google Earth Engine</strong></p>
</div>

---

## 🚀 The Product

### The Problem
Traditional weather forecasting systems rely on numerical modeling that struggles to detect the complex, multi-variable interactions causing extreme climate catastrophes (e.g., compounding heatwaves coupled with severe soil-moisture depletion). Furthermore, when technical bodies release hazard data, it is rarely formatted as actionable, local intelligence that non-technical decision-makers can instantly use.

### The Solution: Climate Intelligence
This platform dynamically analyzes **8 independent environmental variables** across a high-resolution geographic matrix. It builds a 30-day temporal memory of the specific region and uses advanced autoencoder mathematics to identify what "normal" looks like uniquely for that coordinate. When a hazard signature is detected, it does not just trigger an alarm—it uses the **SHAP KernelExplainer** to automatically synthesize a human-readable English briefing explaining *exactly* which variables are causing the anomaly.

Users can type any Indian Pincode or City into the dashboard, and the system geometrically snaps them to the closest AI-monitored matrix grid, presenting a clean, Glassmorphism-styled **Intelligence Summary**. 

---

## 🏗️ Technical Architecture

The architecture scales horizontally across four distinct layers:

### 1. Ingestion Layer (`data/`)
* **Google Earth Engine (GEE)**: Automatically scrapes high-resolution ERA-5 multi-band satellite data (Temp Max, Temp Min, Prec, Wind, Solar, Volumetric Soil Moisture, Leaf Area Index).
* **DuckDB Analytical Engine**: A blazing-fast, in-process columnar database that hosts the `climate_daily` raw tables, pre-computes neural network projections, and caches geocoded `pincode_lookup` spatial hashes for ultra-low latency frontend coordinate resolution (< 5ms).

### 2. Deep Learning Core (`models/`)
* **Spatiotemporal Autoencoder (STAE)**: A Bidirectional-LSTM autoencoder architecture that ingests sequence tensors and calculates a dynamic "Reconstruction Error" matrix. The algorithm autonomously bumps its mathematical thresholds up for naturally high-variance grid cells (e.g. desert margins). 
* **Attention-BiLSTM Forecaster**: An ensemble pipeline consisting of three independent recurrent neural networks equipped with Temporal Attention layers. It outputs 10-day forecasts wrapped in 95% Confidence Intervals mapped via Monte Carlo Dropout uncertainty tracking.
* **Learned Anomaly Topology**: A discrete Random Forest classifier operates on the STAE latent bottleneck layer to statistically categorize the exact physical nature of the detected anomaly (e.g., Drought, Heatwave, Cold Spell).

### 3. Explainer Engine (`shap_explainer`)
* Operates asynchronously, analyzing the flagged anomalous sequence against 50+ background clusters. It derives exact percentage importance vectors and passes them into a heuristic compiler, which outputs a clear, natural-language threat summary (e.g., *"60% driven by irregular patterns in Volumetric Soil Water Layer 1"*).

### 4. API & Presentation Layer (`api/`, `frontend/`)
* **FastAPI Backend**: Orchestrates the communication layer. Completely decouples the Streamlit GUI from the database and ML components, establishing safe, independent endpoints for massive scale up.
* **Streamlit Interface**: Designed with custom responsive CSS, this frontend abandons cluttered technical graphs in favor of a minimalist "Briefing Dashboard", only utilizing graphs to establish physical context (e.g. historic temperature severity correlation tracking).

---

## 💼 Cross-Industry Product Usecases

While this suite was originally designed for systemic academic tracking, its REST API decoupling immediately translates to immense multi-vertical product value:

#### 🚜 AgriTech & Crop Insurance
* **Dynamic Premium Modeling**: Crop insurers can ping the `/api/v1/forecast/latest/{grid_id}` endpoint to automatically adjust regional insurance premium underwriting models based on the 10-day ensemble risk probabilities directly tied to the soil moisture metrics.
* **Precision Harvest Alerts**: SMS integrations pulling from the `/api/v1/anomaly/briefing` endpoint can directly mass-broadcast automated native-language alerts to registered farmers containing specific early-warning insights, securing supply chains days before disaster strikes.

#### 🚢 Smart Logistics & Supply Chain
* **Route Recalibrating Algorithms**: Enterprise logistical software can ingest the geographic anomaly risk heatmaps. If an "Extreme Rainfall/Flooding" risk registers across a 400km highway matrix, algorithms can reroute delivery fleets autonomously to bypass systemic delays.

#### 🏛️ Civic Planning & Emergency Response
* Emergency Response Agencies can bypass data bloat entirely. Rather than monitoring 25 scattered sensor arrays, mayors and district officials just login to local `localhost:8501`, type their Pincode, and receive a red/yellow/green severity light reading directly mapped to exactly what resource is spiking (heat, rain, etc.) so they can deploy aid preemptively.

---

## 💻 Quick Start & Installation

**Prerequisites:** Python 3.10+, pip, venv.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Climate-Intelligence-Platform.git
cd Climate-Intelligence-Platform

# 2. Setup your virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Initialize Database Pipeline & Start Servers
chmod +x run.sh
./run.sh --init
```
*(Note: `--init` wipes your local DuckDB and re-pulls historical GEE context data)*. The system will boot the API on `:8000` and the Dashboard on `:8501`.

### Background Training
The `scripts/train_all.py` script traverses all 25 grid matrices, updating base reconstruction weights and synthesizing SHAP explanations sequentially without locking the API threads.

```bash
nohup python3 scripts/train_all.py --skip-data-check > logs/training.log 2>&1 &
```

---

## 🛣️ Future Growth Roadmap
1. **AWS PostGIS Spatial Scaling**: Transition the memory-cached DuckDB spatial geometry mapping engine into a remote Amazon Aurora PostGIS instance, easily scaling from 25 localized points to an 18,000-matrix grid covering the entire Eastern Hemisphere with 5km granularity.
2. **LLM Explainer Upgrade**: Transition the heuristic SHAP string templating into an open-source `Llama-3` API inference pipeline, passing the SHAP vectors as a JSON prefix prompt to generate highly conversational, natively translated local intelligence reports.
3. **Webhook Subscriptions**: Implement FastAPI reverse-webhook architecture, allowing 3rd party insurers or apps to POST subscription payloads and receive immediate push notifications when their registered grid coordinates cross the `[High Severity]` threshold.

---
<div align="center">
  <small><em>Designed and coded entirely as a standalone AI agentic architecture prototype. Let's build a safer future.</em></small>
</div>

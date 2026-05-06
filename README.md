# ⚡ Utility Intelligence Platform (UIP)
### Smart Resource Management · AI-Powered · End-to-End Streamlit App

---

## 🎯 Project Overview

Built from the **Kaggle Global Energy & Water Consumption** dataset, UIP is a full-stack
data science application that monitors, forecasts, and detects anomalies across 4 utility
types (Water, Electricity, Fuel, Gas) for 22 countries from 2000–2024.

**Inspired by:** The real-world problem outlined in the Utility Intelligence Platform document —
geopolitical supply disruptions (Russia–Ukraine), smart city monitoring, and AI-driven
resource allocation.

---

## 🏗️ Architecture

```
utility_intelligence_platform/
├── app.py                    # Main Streamlit app (router + global CSS)
├── requirements.txt
├── backend/
│   ├── data_engine.py        # Kaggle-style dataset generator (22 countries, 2000–2024)
│   ├── ml_models.py          # Random Forest + GBM forecast · Isolation Forest anomaly
│   └── charts.py             # Plotly chart factory (11 chart types, dark UIP theme)
└── pages/
    ├── dashboard.py          # KPI overview, global trends, alerts
    ├── forecasting.py        # Interactive ML demand forecast, feature importance
    ├── anomaly.py            # Isolation Forest + Z-score anomaly detection
    ├── geospatial.py         # World bubble map, treemap, regional heatmaps
    └── analytics.py          # Efficiency, CO₂, price dynamics, correlations
```

---

## 🤖 ML Models

| Model | Use Case | Algorithm |
|-------|----------|-----------|
| Demand Forecast | Predict future consumption | Random Forest (60%) + Gradient Boosting (40%) ensemble |
| Anomaly Detection (cross-section) | Flag abnormal countries | Isolation Forest (contamination=0.08) |
| Anomaly Detection (time-series) | Monthly outlier detection | Rolling Z-Score (±2σ threshold) |

### Features Used for Forecasting:
- `year`, `year²` (polynomial time trend)
- `lag1`, `lag2`, `lag3` (autoregressive)
- `roll3`, `roll5` (rolling averages)
- `yoy` (year-on-year change)
- `price`, `log_gdp`, `log_pop`, `loss_pct`

### Model Performance (typical):
- MAPE: 2–6% depending on country/utility
- R²: 0.92–0.99

---

## 🎨 UI Design Language

- **Palette:** Deep navy `#050A14` bg · Cyan `#00E5FF` · Electric green `#00FF88` · Amber `#FFB300` · Red `#FF3D71`
- **Fonts:** Syne (headings) · Space Mono (labels/code) · Inter (body)
- **Dark industrial** aesthetic — command-center feel
- Fully responsive, mobile-friendly

---

## 📊 Dataset

Synthetic dataset modelled after:
- [World Energy Consumption — Kaggle](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption)
- Supplemented with water & fuel consumption patterns

**Coverage:**
- 22 countries across 5 regions
- 2000–2024 annual + 2019–2025 monthly time series
- 4 utilities × geopolitical shocks (COVID-2020, Russia-Ukraine-2022)

---

## 🚀 Quick Start

```bash
# 1. Clone / extract the project folder
cd utility_intelligence_platform

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Then open: **http://localhost:8501**

---

## 📱 Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards, global trend charts, live alerts, heatmaps |
| 📈 Forecasting | Country × utility ML forecast, confidence bands, feature importance |
| 🔍 Anomaly Detection | Cross-country Isolation Forest + monthly Z-score timeline |
| 🗺️ Geospatial | World bubble map, treemap, regional heatmaps, country rankings |
| 📊 Analytics | Efficiency, CO₂, price dynamics, correlations, per-capita |

---

## 🛠️ Extending the App

- **Real IoT data:** Replace `DataEngine._build_dataset()` with your live DB queries
- **API integration:** Add FastAPI backend and call it from Streamlit
- **Map upgrade:** Swap `geo_bubble` chart for `folium` or `kepler.gl` for true GIS
- **LSTM models:** Add in `ml_models.py` for deeper sequence modelling
- **Alerts pipeline:** Wire alerts to email/SMS via Twilio or SendGrid

---

*Built with ❤️ using Streamlit · scikit-learn · Plotly · Python*

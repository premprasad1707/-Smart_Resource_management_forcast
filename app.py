"""
╔══════════════════════════════════════════════════════════════╗
║        UTILITY INTELLIGENCE PLATFORM                        ║
║        Smart Resource Management · AI-Powered               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Utility Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #050A14;
    --bg-secondary: #0A1628;
    --bg-card: #0D1F3C;
    --bg-card-hover: #112447;
    --accent-cyan: #00E5FF;
    --accent-electric: #00FF88;
    --accent-amber: #FFB300;
    --accent-red: #FF3D71;
    --accent-purple: #7B5EA7;
    --text-primary: #E8F4FD;
    --text-secondary: #8BA3C4;
    --text-muted: #4A6A8A;
    --border: rgba(0, 229, 255, 0.15);
    --glow-cyan: 0 0 30px rgba(0, 229, 255, 0.3);
    --glow-green: 0 0 30px rgba(0, 255, 136, 0.3);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main App Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0A1F3A 0%, #050A14 50%, #020710 100%) !important;
    background-attachment: fixed !important;
}

.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1600px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060D1E 0%, #0A1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 50%, #071120 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(0,229,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}

.hero-header::after {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.4rem !important;
}

.hero-title span {
    background: linear-gradient(90deg, #00E5FF, #00FF88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-electric);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-electric);
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent-electric); }
    50% { opacity: 0.5; box-shadow: 0 0 16px var(--accent-electric); }
}

/* ── KPI Cards ── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: default;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}

.kpi-card.green::before { background: linear-gradient(90deg, var(--accent-electric), transparent); }
.kpi-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.kpi-card.red::before   { background: linear-gradient(90deg, var(--accent-red), transparent); }

.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.6rem;
}

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
}

.kpi-value.cyan  { color: var(--accent-cyan); }
.kpi-value.green { color: var(--accent-electric); }
.kpi-value.amber { color: var(--accent-amber); }
.kpi-value.red   { color: var(--accent-red); }

.kpi-delta {
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

.kpi-delta.up   { color: var(--accent-electric); }
.kpi-delta.down { color: var(--accent-red); }

.kpi-icon {
    position: absolute;
    top: 1.2rem; right: 1.2rem;
    font-size: 1.6rem;
    opacity: 0.2;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
}

/* ── Chart Container ── */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Alert Badges ── */
.alert-item {
    background: rgba(255,61,113,0.08);
    border: 1px solid rgba(255,61,113,0.25);
    border-left: 3px solid var(--accent-red);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

.alert-item.warning {
    background: rgba(255,179,0,0.06);
    border-color: rgba(255,179,0,0.2);
    border-left-color: var(--accent-amber);
}

.alert-item.info {
    background: rgba(0,229,255,0.06);
    border-color: rgba(0,229,255,0.2);
    border-left-color: var(--accent-cyan);
}

.alert-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-red);
    margin-bottom: 0.2rem;
}

.alert-item.warning .alert-title { color: var(--accent-amber); }
.alert-item.info    .alert-title { color: var(--accent-cyan); }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    background: var(--bg-secondary) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label, .stRadio label, .stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
}

button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #00E5FF, #0088CC) !important;
    color: #050A14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover, .stButton > button:hover {
    box-shadow: 0 0 20px rgba(0,229,255,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0A1F3A 0%, #071620 100%);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,229,255,0.1);
}

.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00E5FF, #00FF88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.pred-unit {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Status pill ── */
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    font-weight: 700;
    text-transform: uppercase;
}

.status-pill.critical { background: rgba(255,61,113,0.15); color: #FF3D71; border: 1px solid rgba(255,61,113,0.3); }
.status-pill.warning  { background: rgba(255,179,0,0.12);  color: #FFB300; border: 1px solid rgba(255,179,0,0.3); }
.status-pill.normal   { background: rgba(0,255,136,0.1);   color: #00FF88; border: 1px solid rgba(0,255,136,0.25); }

</style>
""", unsafe_allow_html=True)

# ── Imports ─────────────────────────────────────────────────────
from backend.data_engine import DataEngine
from backend.ml_models import ForecastEngine, AnomalyDetector
from backend.charts import ChartBuilder

# ── Init ─────────────────────────────────────────────────────────
@st.cache_resource
def load_engines():
    data = DataEngine()
    forecast = ForecastEngine(data)
    anomaly = AnomalyDetector(data)
    charts = ChartBuilder()
    return data, forecast, anomaly, charts

data_eng, forecast_eng, anomaly_det, chart_builder = load_engines()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#E8F4FD;letter-spacing:-0.02em;">
            ⚡ UIP
        </div>
        <div style="font-size:0.65rem;color:#4A6A8A;letter-spacing:0.12em;text-transform:uppercase;
                    font-family:'Space Mono',monospace;margin-top:0.2rem;">
            Utility Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.65rem;color:#4A6A8A;letter-spacing:0.1em;text-transform:uppercase;font-family:\'Space Mono\',monospace;margin-bottom:0.5rem;">Navigation</p>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "📈  Forecasting", "🔍  Anomaly Detection", "🗺️  Geospatial", "📊  Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.65rem;color:#4A6A8A;letter-spacing:0.1em;text-transform:uppercase;font-family:\'Space Mono\',monospace;margin-bottom:0.5rem;">Filters</p>', unsafe_allow_html=True)

    utility_type = st.selectbox("Resource Type", ["All", "Water", "Electricity", "Fuel", "Gas"])
    region = st.selectbox("Region", ["Global", "North America", "Europe", "Asia Pacific", "Middle East", "Africa"])
    year_range = st.slider("Year Range", 2015, 2024, (2019, 2024))

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem;color:#2A4A6A;font-family:'Space Mono',monospace;line-height:1.7;">
        <div><span style="color:#00E5FF;">●</span> MODEL: Random Forest + XGBoost</div>
        <div><span style="color:#00FF88;">●</span> DATA: Kaggle Global Utility</div>
        <div><span style="color:#FFB300;">●</span> LAST SYNC: Live</div>
    </div>
    """, unsafe_allow_html=True)

# ── Route Pages ──────────────────────────────────────────────────
if   page == "🏠  Dashboard":        from pages import dashboard;   dashboard.render(data_eng, forecast_eng, anomaly_det, chart_builder, utility_type, region, year_range)
elif page == "📈  Forecasting":       from pages import forecasting; forecasting.render(data_eng, forecast_eng, chart_builder, utility_type, region)
elif page == "🔍  Anomaly Detection": from pages import anomaly;     anomaly.render(data_eng, anomaly_det, chart_builder, utility_type, region)
elif page == "🗺️  Geospatial":        from pages import geospatial;  geospatial.render(data_eng, chart_builder, utility_type, region)
elif page == "📊  Analytics":         from pages import analytics;   analytics.render(data_eng, chart_builder, utility_type, region, year_range)

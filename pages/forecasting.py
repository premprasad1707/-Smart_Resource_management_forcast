"""
Forecasting Page – ML-powered demand prediction
"""
import streamlit as st
import pandas as pd
import numpy as np


def render(data_eng, forecast_eng, chart_builder, utility_type, region):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📈 Demand <span>Forecasting</span></div>
        <div class="hero-subtitle">Random Forest + Gradient Boosting Ensemble · 5-Year Outlook</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_country = st.selectbox(
            "Select Country",
            data_eng.countries,
            index=data_eng.countries.index("USA") if "USA" in data_eng.countries else 0,
        )
    with c2:
        selected_utility = st.selectbox(
            "Select Utility",
            ["Electricity", "Water", "Fuel", "Gas"],
            index=0,
        )
    with c3:
        forecast_years = st.slider("Forecast Horizon (Years)", 1, 10, 5)

    # ── Run Forecast ─────────────────────────────────────────────
    result = forecast_eng.predict(selected_country, selected_utility, forecast_years)

    if result is None:
        st.warning("Insufficient data for this combination.")
        return

    # ── Model Metrics Row ────────────────────────────────────────
    m = result["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, unit, color in [
        (mc1, "MAPE",  f"{m['mape']:.2f}",  "%",    "cyan"),
        (mc2, "R² Score", f"{m['r2']:.3f}", "",     "green"),
        (mc3, "RMSE",  f"{m['rmse']:.0f}",  "TWh",  "amber"),
        (mc4, "Model", "RF+GBM",            "Ensemble", "cyan"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {color}" style="font-size:1.6rem;">{val}</div>
                <div style="font-size:0.7rem;color:#4A6A8A;">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Main Forecast Chart ──────────────────────────────────────
    st.markdown(f'<div class="section-header">🔮 {selected_country} · {selected_utility} Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Historical fit + ML forecast with 90% confidence interval</div>', unsafe_allow_html=True)

    fig = chart_builder.forecast_chart(result)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Values Table + Feature Importance ───────────────
    col_table, col_fi = st.columns([1, 1])

    with col_table:
        st.markdown('<div class="section-header">📋 Forecast Values</div>', unsafe_allow_html=True)
        df_fut = result["df"][result["df"]["type"] == "forecast"][
            ["year", "predicted", "lower_90", "upper_90"]].copy()
        df_fut.columns = ["Year", "Predicted", "Lower 90%", "Upper 90%"]
        df_fut = df_fut.round(1)

        # Style the table
        st.markdown("""
        <style>
        .stDataFrame { background: transparent !important; }
        .stDataFrame table { border-collapse: collapse; width: 100%; }
        .stDataFrame th {
            background: #0A1628 !important; color: #4A6A8A !important;
            font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important;
            text-transform: uppercase; letter-spacing: 0.06em; padding: 0.5rem 0.8rem;
            border-bottom: 1px solid rgba(0,229,255,0.1);
        }
        .stDataFrame td {
            color: #E8F4FD !important; font-family: 'Space Mono', monospace !important;
            font-size: 0.8rem !important; padding: 0.4rem 0.8rem;
            border-bottom: 1px solid rgba(0,229,255,0.05) !important;
            background: #0D1F3C !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(df_fut, use_container_width=True, hide_index=True)

    with col_fi:
        st.markdown('<div class="section-header">🧠 Feature Importance</div>', unsafe_allow_html=True)
        fig_fi = chart_builder.feature_importance(
            result["feature_importance"],
            "Which features drive the model predictions?"
        )
        fig_fi.update_layout(height=300)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Bulk Forecast: all countries ─────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-header">🌍 Global {selected_utility} Demand Forecast (All Countries)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Projected demand 3 years forward across all monitored nations</div>', unsafe_allow_html=True)

    with st.spinner("Computing bulk forecast..."):
        bulk = forecast_eng.bulk_forecast(selected_utility, forecast_years=3)
        if not bulk.empty:
            pivot = bulk.pivot_table(index="country", columns="year",
                                     values="predicted", aggfunc="sum").round(1)
            st.dataframe(pivot, use_container_width=True)

    # ── Model Summary ─────────────────────────────────────────────
    with st.expander("🔬 View All Model Performance Metrics"):
        summary = forecast_eng.get_model_summary()
        summary_sel = summary[summary["Utility"] == selected_utility].reset_index(drop=True)
        st.dataframe(summary_sel, use_container_width=True, hide_index=True)

    # ── Forecast Insight Box ──────────────────────────────────────
    fut_df = result["df"][result["df"]["type"] == "forecast"]
    last_hist = result["df"][result["df"]["type"] == "historical"]["actual"].iloc[-1]
    last_fore = fut_df["predicted"].iloc[-1]
    change_pct = (last_fore - last_hist) / last_hist * 100

    direction = "📈 INCREASE" if change_pct > 0 else "📉 DECREASE"
    color_dir = "#FF3D71" if change_pct > 5 else "#00FF88" if change_pct <= 0 else "#FFB300"

    st.markdown(f"""
    <div class="pred-box" style="margin-top:1.5rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#4A6A8A;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">
            {forecast_years}-YEAR OUTLOOK · {selected_country} · {selected_utility}
        </div>
        <div class="pred-value">{last_fore:,.0f}</div>
        <div class="pred-unit">Projected {selected_utility} Units in {2024 + forecast_years}</div>
        <div style="margin-top:0.8rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:{color_dir};">
            {direction} of {abs(change_pct):.1f}% from current baseline
        </div>
        <div style="font-size:0.72rem;color:#4A6A8A;margin-top:0.4rem;">
            Model confidence: R² = {m['r2']:.3f} · MAPE = {m['mape']:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

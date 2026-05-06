"""
Dashboard Page – Real-time overview, KPIs, global trends, alerts
"""
import streamlit as st
import pandas as pd
import numpy as np


def render(data_eng, forecast_eng, anomaly_det, chart_builder, utility_type, region, year_range):
    # ── Hero Header ──────────────────────────────────────────────
    now = "2024 · Q4"
    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
            <div>
                <div class="hero-title">Utility Intelligence <span>Platform</span></div>
                <div class="hero-subtitle">Smart Resource Management · AI-Powered Command Center</div>
                <div style="margin-top:1rem;">
                    <span class="hero-badge">⚡ ELECTRICITY</span>
                    <span class="hero-badge">💧 WATER</span>
                    <span class="hero-badge">🛢️ FUEL</span>
                    <span class="hero-badge">🔥 GAS</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#4A6A8A;margin-bottom:0.5rem;">
                    SYSTEM STATUS
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#00FF88;margin-bottom:0.3rem;">
                    <span class="live-dot"></span> ALL SYSTEMS ONLINE
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#2A4A6A;">
                    PERIOD: {now} · {region} · ML MODELS ACTIVE
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────
    kpis = data_eng.get_kpis()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_configs = [
        (c1, "Total Consumption",    f"{kpis['total_consumption']/1e6:.2f}M",  "TWh + Km³ equiv",
         f"▲ {kpis['consumption_delta']:.1f}% YoY", "up", "cyan",  "⚡"),
        (c2, "Countries Monitored",  str(kpis["countries"]),                   "Nations tracked",
         "▲ 5 new in 2024", "up", "green", "🌐"),
        (c3, "Shortage Risk Zones",  f"{kpis['shortage_risk_pct']:.1f}%",      "of country-utility pairs",
         "▲ 2.3% vs last yr", "down", "red",   "⚠️"),
        (c4, "Avg Distribution Loss",f"{kpis['avg_loss_pct']:.1f}%",          "of total supply wasted",
         "▼ 0.8% improved", "up", "amber", "📉"),
        (c5, "CO₂ Equivalent",       f"{kpis['co2_mn_tonnes']:.0f}Mn",         "Tonnes CO₂ (2024)",
         "▼ 1.2% vs 2023", "up", "cyan",  "🌿"),
    ]

    for col, label, value, unit, delta, d_dir, color, icon in kpi_configs:
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {color}">{value}</div>
                <div style="font-size:0.68rem;color:#4A6A8A;margin-bottom:0.3rem;">{unit}</div>
                <div class="kpi-delta {'up' if d_dir=='up' and color!='red' else 'down'}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # ── Row 2: Trend + Alert Panel ───────────────────────────────
    col_chart, col_alerts = st.columns([2.5, 1])

    with col_chart:
        st.markdown('<div class="section-header">📈 Global Consumption Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Annual consumption by resource type · 2000–2024</div>', unsafe_allow_html=True)

        df = data_eng.get_annual(utility_type, region, year_range)
        trend = df.groupby(["year", "utility"])["consumption"].sum().reset_index()

        fig = chart_builder.time_series_multi(trend, "year", "consumption", "utility",
                                               "Global Resource Consumption by Utility Type")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_alerts:
        st.markdown('<div class="section-header">🚨 Active Alerts</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Critical & warning events · 2024</div>', unsafe_allow_html=True)

        alerts = data_eng.get_alerts()
        for a in alerts[:6]:
            sev = a["severity"]
            st.markdown(f"""
            <div class="alert-item {sev}">
                <div class="alert-title">{sev.upper()} · {a['utility']}</div>
                <div>{a['country']} — {a['message']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 3: Bar + Donut ───────────────────────────────────────
    col_bar, col_donut = st.columns([2, 1])

    with col_bar:
        st.markdown('<div class="section-header">🏆 Top Countries by Consumption (2024)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Total resource consumption across all utilities</div>', unsafe_allow_html=True)
        cs = data_eng.get_country_summary(2024).nlargest(12, "total_consumption")
        fig2 = chart_builder.bar_chart(cs, x="country", y="total_consumption",
                                        title="Top 12 Countries – Total Consumption")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with col_donut:
        st.markdown('<div class="section-header">🥧 Mix by Utility</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">2024 global share</div>', unsafe_allow_html=True)
        mix = data_eng.df[data_eng.df["year"] == 2024].groupby("utility")["consumption"].sum()
        fig3 = chart_builder.donut(mix.index.tolist(), mix.values.tolist(), "Utility Mix")
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 4: Heatmap ───────────────────────────────────────────
    st.markdown('<div class="section-header">🔥 Region × Utility Consumption Heatmap (2024)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Intensity = total consumption per region-utility pair</div>', unsafe_allow_html=True)

    heat_df = data_eng.df[data_eng.df["year"] == 2024].groupby(
        ["region", "utility"])["consumption"].sum().reset_index()
    fig4 = chart_builder.heatmap(heat_df, "utility", "region", "consumption",
                                  "Regional Consumption Heatmap (2024)")
    fig4.update_layout(height=280)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Row 5: Loss % + Shortage events ─────────────────────────
    col_loss, col_short = st.columns(2)

    with col_loss:
        st.markdown('<div class="section-header">📉 Distribution Loss % by Country</div>', unsafe_allow_html=True)
        cs_loss = data_eng.get_country_summary(2024).nlargest(15, "avg_loss_pct")
        fig5 = chart_builder.bar_chart(cs_loss, x="avg_loss_pct", y="country",
                                        title="Average Distribution Loss % (2024)",
                                        orientation="h")
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)

    with col_short:
        st.markdown('<div class="section-header">⚡ Shortage Events by Year</div>', unsafe_allow_html=True)
        short_yr = data_eng.df.groupby(["year","utility"])["shortage_risk"].sum().reset_index()
        fig6 = chart_builder.time_series_multi(short_yr, "year", "shortage_risk", "utility",
                                               "Annual Shortage Events per Utility")
        fig6.update_layout(height=350)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#2A4A6A;
                text-align:center;padding:0.5rem;">
        UTILITY INTELLIGENCE PLATFORM · ML MODELS: Random Forest + Gradient Boosting · 
        ANOMALY DETECTION: Isolation Forest · DATA: Kaggle Global Utility Dataset
    </div>
    """, unsafe_allow_html=True)

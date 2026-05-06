"""
Anomaly Detection Page – Isolation Forest + Z-Score
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render(data_eng, anomaly_det, chart_builder, utility_type, region):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔍 Anomaly <span>Detection</span></div>
        <div class="hero-subtitle">Isolation Forest · Statistical Z-Score · Real-time Outlier Identification</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        util_sel = st.selectbox("Utility Type", ["Electricity", "Water", "Fuel", "Gas"])
    with col2:
        year_sel = st.selectbox("Year", list(range(2024, 2014, -1)))
    with col3:
        ts_country = st.selectbox(
            "Country (Time-Series)",
            ["USA", "China", "Germany", "India", "Saudi Arabia", "Nigeria"],
        )

    # ── Cross-section Anomalies ───────────────────────────────────
    st.markdown(f'<div class="section-header">🌐 Cross-Country Anomaly Detection · {util_sel} · {year_sel}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Isolation Forest scoring across all country observations</div>', unsafe_allow_html=True)

    df_anom = anomaly_det.detect(util_sel, year_sel)

    if df_anom.empty:
        st.info("No anomaly data available.")
        return

    # KPIs
    n_anomalies = df_anom["anomaly"].sum()
    pct = n_anomalies / len(df_anom) * 100
    max_score = df_anom["anomaly_score"].max()
    high_risk = df_anom[df_anom["severity"] == "High"]["country"].tolist()

    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val, unit, clr in [
        (k1, "Anomalies Detected", str(n_anomalies), f"of {len(df_anom)} countries", "red"),
        (k2, "Detection Rate",     f"{pct:.1f}%",    "contamination rate",           "amber"),
        (k3, "Max Risk Score",     f"{max_score:.3f}", "Isolation Forest",            "red"),
        (k4, "High-Risk Countries", str(len(high_risk)), "severity = HIGH",           "amber"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {clr}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value {clr}" style="font-size:1.6rem;">{val}</div>
                <div style="font-size:0.7rem;color:#4A6A8A;">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Bubble Chart ──────────────────────────────────────────────
    col_bub, col_list = st.columns([2, 1])

    with col_bub:
        fig_bub = chart_builder.bubble_chart(
            df_anom, x="consumption", y="loss_pct",
            size="anomaly_score", color_col="severity",
            title="Anomaly Map: Consumption vs Loss% (bubble = risk score)"
        )
        fig_bub.update_layout(height=380)
        st.plotly_chart(fig_bub, use_container_width=True)

    with col_list:
        st.markdown('<div class="section-header">⚠️ Flagged Countries</div>', unsafe_allow_html=True)
        flagged = df_anom[df_anom["anomaly"] == 1][
            ["country", "anomaly_score", "severity", "consumption", "loss_pct"]
        ].head(10)

        for _, row in flagged.iterrows():
            sev = str(row["severity"]).lower()
            badge_class = "critical" if sev == "high" else "warning" if sev == "medium" else "normal"
            st.markdown(f"""
            <div class="alert-item {'critical' if sev=='high' else 'warning'}">
                <div class="alert-title">{row['country']}</div>
                <div>Score: <b>{row['anomaly_score']:.3f}</b> · Loss: {row['loss_pct']:.1f}%</div>
                <div><span class="status-pill {badge_class}">{str(row['severity']).upper()}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Time-Series Anomaly ───────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-header">📅 Monthly Anomaly Timeline · {ts_country} · {util_sel}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Rolling Z-Score method · threshold = ±2σ</div>', unsafe_allow_html=True)

    df_ts = anomaly_det.time_series_anomaly(ts_country, util_sel)
    if df_ts.empty:
        st.info("No time-series data for this combination.")
    else:
        fig_ts = chart_builder.anomaly_timeline(df_ts,
            title=f"{ts_country} · {util_sel} Monthly Consumption Anomalies")
        fig_ts.update_layout(height=340)
        st.plotly_chart(fig_ts, use_container_width=True)

        # Stats
        n_ts_anom = df_ts["anomaly"].sum()
        spikes = (df_ts["anomaly"] == 1) & (df_ts["direction"] == "spike")
        dips   = (df_ts["anomaly"] == 1) & (df_ts["direction"] == "dip")

        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            st.markdown(f"""<div class="kpi-card red">
                <div class="kpi-label">Total Anomalies</div>
                <div class="kpi-value red" style="font-size:1.6rem;">{n_ts_anom}</div>
                <div style="font-size:0.7rem;color:#4A6A8A;">monthly outliers</div></div>""",
                unsafe_allow_html=True)
        with sa2:
            st.markdown(f"""<div class="kpi-card amber">
                <div class="kpi-label">Demand Spikes</div>
                <div class="kpi-value amber" style="font-size:1.6rem;">{spikes.sum()}</div>
                <div style="font-size:0.7rem;color:#4A6A8A;">above +2σ</div></div>""",
                unsafe_allow_html=True)
        with sa3:
            st.markdown(f"""<div class="kpi-card cyan">
                <div class="kpi-label">Demand Dips</div>
                <div class="kpi-value cyan" style="font-size:1.6rem;">{dips.sum()}</div>
                <div style="font-size:0.7rem;color:#4A6A8A;">below -2σ</div></div>""",
                unsafe_allow_html=True)

    # ── Z-Score Distribution ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Anomaly Score Distribution</div>', unsafe_allow_html=True)

    if not df_anom.empty:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig_dist = make_subplots(rows=1, cols=2,
            subplot_titles=["Isolation Forest Score Distribution", "Z-Score Distribution"])

        fig_dist.add_trace(go.Histogram(
            x=df_anom["anomaly_score"], nbinsx=20,
            marker_color="#00E5FF", opacity=0.7,
            name="Anomaly Score",
        ), row=1, col=1)

        fig_dist.add_trace(go.Histogram(
            x=df_anom["z_consumption"], nbinsx=20,
            marker_color="#00FF88", opacity=0.7,
            name="Z-Score",
        ), row=1, col=2)

        fig_dist.update_layout(
            paper_bgcolor="#0D1F3C", plot_bgcolor="#070F20",
            font=dict(color="#E8F4FD", size=11),
            margin=dict(l=16, r=16, t=40, b=16),
            showlegend=False, height=280,
        )
        fig_dist.update_xaxes(gridcolor="rgba(0,229,255,0.06)",
                               tickfont=dict(color="#4A6A8A"))
        fig_dist.update_yaxes(gridcolor="rgba(0,229,255,0.06)",
                               tickfont=dict(color="#4A6A8A"))
        st.plotly_chart(fig_dist, use_container_width=True)

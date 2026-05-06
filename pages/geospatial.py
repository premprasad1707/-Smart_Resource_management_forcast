"""
Geospatial Page – World map, regional breakdowns, GIS-style visualisations
"""
import streamlit as st
import pandas as pd


def render(data_eng, chart_builder, utility_type, region):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🗺️ <span>Geospatial</span> Intelligence</div>
        <div class="hero-subtitle">GIS-Powered Resource Mapping · Global Distribution Visibility</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        util_map = st.selectbox("Utility for Map", ["Electricity", "Water", "Fuel", "Gas"], key="geo_util")
    with col2:
        year_map = st.selectbox("Year", list(range(2024, 2014, -1)), key="geo_year")

    # ── Global Bubble Map ─────────────────────────────────────────
    st.markdown('<div class="section-header">🌍 Global Resource Distribution Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Bubble size = consumption volume · Color = distribution loss %</div>', unsafe_allow_html=True)

    cs = data_eng.get_country_summary(year_map)
    # Filter by utility
    util_df = data_eng.df[
        (data_eng.df["year"] == year_map) &
        (data_eng.df["utility"] == util_map)
    ][["country", "consumption", "loss_pct", "shortage_risk", "region"]].copy()

    if not util_df.empty:
        fig_geo = chart_builder.geo_bubble(
            util_df, size_col="consumption", color_col="loss_pct",
            title=f"{util_map} – Global Consumption & Distribution Loss ({year_map})"
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    # ── Regional Breakdown ────────────────────────────────────────
    st.markdown("---")
    col_reg, col_tree = st.columns(2)

    with col_reg:
        st.markdown('<div class="section-header">📊 Consumption by Region</div>', unsafe_allow_html=True)
        reg_df = data_eng.df[
            (data_eng.df["year"] == year_map) &
            (data_eng.df["utility"] == util_map)
        ].groupby("region")["consumption"].sum().reset_index()
        reg_df = reg_df.sort_values("consumption", ascending=True)

        fig_reg = chart_builder.bar_chart(
            reg_df, x="consumption", y="region",
            title=f"{util_map} Consumption by Region ({year_map})",
            orientation="h"
        )
        fig_reg.update_layout(height=320)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_tree:
        st.markdown('<div class="section-header">🌳 Consumption Treemap</div>', unsafe_allow_html=True)
        tree_df = data_eng.df[
            (data_eng.df["year"] == year_map) &
            (data_eng.df["utility"] == util_map)
        ][["region", "country", "consumption"]].copy()

        if not tree_df.empty:
            fig_tree = chart_builder.treemap(
                tree_df, path_cols=["region", "country"],
                values="consumption",
                title=f"{util_map} Consumption Hierarchy ({year_map})"
            )
            fig_tree.update_layout(height=320)
            st.plotly_chart(fig_tree, use_container_width=True)

    # ── Shortage Risk Map ─────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⚠️ Shortage Risk Heatmap · All Utilities vs Regions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Number of shortage events detected in the selected year</div>', unsafe_allow_html=True)

    risk_df = data_eng.df[data_eng.df["year"] == year_map].groupby(
        ["region", "utility"])["shortage_risk"].sum().reset_index()

    fig_risk = chart_builder.heatmap(
        risk_df, x="utility", y="region", z="shortage_risk",
        title=f"Shortage Event Count by Region × Utility ({year_map})"
    )
    fig_risk.update_layout(height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

    # ── Country Rankings Table ────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-header">📋 Country Rankings · {util_map} · {year_map}</div>', unsafe_allow_html=True)

    rank_df = data_eng.df[
        (data_eng.df["year"] == year_map) &
        (data_eng.df["utility"] == util_map)
    ][["country", "region", "consumption", "supply", "gap",
       "loss_pct", "per_capita", "co2", "shortage_risk"]].copy()

    rank_df = rank_df.sort_values("consumption", ascending=False).reset_index(drop=True)
    rank_df.index += 1
    rank_df.columns = [
        "Country", "Region", "Consumption", "Supply", "Gap",
        "Loss %", "Per Capita", "CO₂", "Shortage"
    ]
    rank_df = rank_df.round(2)

    # Highlight shortage rows
    def highlight_shortage(row):
        return ["background-color: rgba(255,61,113,0.08)" if row["Shortage"] == 1
                else "" for _ in row]

    st.dataframe(
        rank_df.style.apply(highlight_shortage, axis=1),
        use_container_width=True,
        height=400,
    )

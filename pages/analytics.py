"""
Analytics Page – Deep-dive: efficiency, CO₂, price trends, correlations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


COLORS = {
    "bg_card": "#0D1F3C", "bg_plot": "#070F20",
    "grid": "rgba(0,229,255,0.06)", "text": "#E8F4FD",
    "text_muted": "#4A6A8A", "cyan": "#00E5FF",
    "green": "#00FF88", "amber": "#FFB300", "red": "#FF3D71",
}


def render(data_eng, chart_builder, utility_type, region, year_range):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Deep <span>Analytics</span></div>
        <div class="hero-subtitle">Energy Intensity · CO₂ Correlation · Price Dynamics · Efficiency Benchmarking</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["⚡ Efficiency", "🌿 CO₂ & Emissions", "💰 Price Dynamics", "🔗 Correlations", "📈 Per-Capita Trends"])

    df_all = data_eng.get_annual(utility_type, region, year_range)

    # ── Tab 1: Efficiency ─────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">Energy Intensity (Consumption / GDP)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Lower = more efficient economy · units per $Bn GDP</div>', unsafe_allow_html=True)

        intens = df_all.groupby(["year", "region"])["intensity"].mean().reset_index()
        fig_intens = chart_builder.time_series_multi(
            intens, "year", "intensity", "region",
            "Average Resource Intensity by Region (2015–2024)"
        )
        fig_intens.update_layout(height=340)
        st.plotly_chart(fig_intens, use_container_width=True)

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Loss % Over Time</div>', unsafe_allow_html=True)
            loss_yr = df_all.groupby(["year","utility"])["loss_pct"].mean().reset_index()
            fig_loss = chart_builder.time_series_multi(
                loss_yr, "year", "loss_pct", "utility",
                "Average Distribution Loss % by Utility"
            )
            fig_loss.update_layout(height=300)
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Supply vs Demand Gap</div>', unsafe_allow_html=True)
            gap_yr = df_all.groupby(["year","utility"])["gap"].mean().reset_index()
            fig_gap = chart_builder.time_series_multi(
                gap_yr, "year", "gap", "utility",
                "Average Supply-Demand Gap by Utility"
            )
            fig_gap.update_layout(height=300)
            st.plotly_chart(fig_gap, use_container_width=True)

        # Efficiency ranking
        st.markdown("---")
        st.markdown('<div class="section-header">🏆 Country Efficiency Ranking (2024)</div>', unsafe_allow_html=True)
        eff = data_eng.df[data_eng.df["year"] == 2024].groupby("country")["intensity"].mean().reset_index()
        eff = eff.sort_values("intensity").head(15)
        fig_eff = chart_builder.bar_chart(eff, x="intensity", y="country",
                                           title="Lowest Intensity = Most Efficient (2024)",
                                           orientation="h")
        fig_eff.update_layout(height=380)
        st.plotly_chart(fig_eff, use_container_width=True)

    # ── Tab 2: CO₂ ────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header">🌿 CO₂ Emissions Trend</div>', unsafe_allow_html=True)
        co2_trend = df_all.groupby(["year","utility"])["co2"].sum().reset_index()
        fig_co2 = chart_builder.time_series_multi(
            co2_trend, "year", "co2", "utility",
            "Total CO₂ Equivalent Emissions by Utility Type"
        )
        fig_co2.update_layout(height=340)
        st.plotly_chart(fig_co2, use_container_width=True)

        col_co2l, col_co2r = st.columns(2)

        with col_co2l:
            st.markdown('<div class="section-header">Top CO₂ Emitters (2024)</div>', unsafe_allow_html=True)
            co2_c = data_eng.df[data_eng.df["year"] == 2024].groupby("country")["co2"].sum().reset_index()
            co2_c = co2_c.nlargest(12, "co2")
            fig_co2c = chart_builder.bar_chart(co2_c, x="country", y="co2",
                                                title="Top 12 CO₂ Emitters (2024)")
            fig_co2c.update_layout(height=320)
            st.plotly_chart(fig_co2c, use_container_width=True)

        with col_co2r:
            st.markdown('<div class="section-header">CO₂ per Utility Type (2024)</div>', unsafe_allow_html=True)
            co2_u = data_eng.df[data_eng.df["year"] == 2024].groupby("utility")["co2"].sum()
            fig_co2u = chart_builder.donut(
                co2_u.index.tolist(), co2_u.values.tolist(), "CO₂ Share"
            )
            fig_co2u.update_layout(height=320)
            st.plotly_chart(fig_co2u, use_container_width=True)

        # CO2 vs consumption scatter
        st.markdown("---")
        st.markdown('<div class="section-header">CO₂ vs Consumption (2024)</div>', unsafe_allow_html=True)
        scatter_df = data_eng.df[data_eng.df["year"] == 2024]
        fig_scat = chart_builder.bubble_chart(
            scatter_df, x="consumption", y="co2",
            size="pop_million", color_col="utility",
            title="CO₂ vs Consumption – bubble size = population (2024)"
        )
        fig_scat.update_layout(height=380)
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Tab 3: Price Dynamics ─────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">💰 Price Trends by Utility</div>', unsafe_allow_html=True)
        price_trend = df_all.groupby(["year","utility"])["price"].mean().reset_index()
        fig_price = chart_builder.time_series_multi(
            price_trend, "year", "price", "utility",
            "Average Utility Price Index Over Time"
        )
        fig_price.update_layout(height=340)
        st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("---")
        col_pl, col_pr = st.columns(2)

        with col_pl:
            st.markdown('<div class="section-header">Price Heatmap: Country × Year</div>', unsafe_allow_html=True)
            sel_u = st.selectbox("Utility", ["Electricity","Gas","Fuel","Water"], key="price_util")
            ph_df = data_eng.df[data_eng.df["utility"] == sel_u].groupby(
                ["country","year"])["price"].mean().reset_index()
            ph_df = ph_df[ph_df["country"].isin(
                ["USA","China","Germany","India","Japan","UK","France","Saudi Arabia"])]
            fig_ph = chart_builder.heatmap(ph_df, "year", "country", "price",
                                            f"{sel_u} Price Heatmap ($/unit)")
            fig_ph.update_layout(height=340)
            st.plotly_chart(fig_ph, use_container_width=True)

        with col_pr:
            st.markdown('<div class="section-header">Price vs Loss Correlation</div>', unsafe_allow_html=True)
            fig_pvl = chart_builder.bubble_chart(
                df_all.sample(min(500, len(df_all))),
                x="price", y="loss_pct", size="consumption",
                color_col="utility",
                title="Does higher price = lower loss? (sample)"
            )
            fig_pvl.update_layout(height=340)
            st.plotly_chart(fig_pvl, use_container_width=True)

    # ── Tab 4: Correlations ───────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Pearson correlation between key numeric variables</div>', unsafe_allow_html=True)

        num_cols = ["consumption", "supply", "gap", "loss_pct", "price",
                    "co2", "per_capita", "intensity", "gdp_bn", "pop_million"]
        corr = data_eng.df[num_cols].corr().round(2)

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, COLORS["red"]], [0.5, COLORS["bg_plot"]], [1.0, COLORS["cyan"]]
            ],
            zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            showscale=True,
            hovertemplate="%{y} vs %{x}: <b>%{z}</b><extra></extra>",
        ))
        fig_corr.update_layout(
            paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_plot"],
            font=dict(color=COLORS["text"], size=10),
            margin=dict(l=10, r=10, t=30, b=10), height=420,
        )
        fig_corr.update_xaxes(tickangle=-45, tickfont=dict(size=9, color=COLORS["text_muted"]))
        fig_corr.update_yaxes(tickfont=dict(size=9, color=COLORS["text_muted"]))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 5: Per-Capita ─────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">👤 Per-Capita Consumption Trends</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Consumption per million population – normalised comparison</div>', unsafe_allow_html=True)

        pc_df = data_eng.df.groupby(["year","utility"])["per_capita"].mean().reset_index()
        fig_pc = chart_builder.time_series_multi(
            pc_df, "year", "per_capita", "utility",
            "Average Per-Capita Consumption by Utility Type"
        )
        fig_pc.update_layout(height=340)
        st.plotly_chart(fig_pc, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Top Per-Capita Consumers (2024)</div>', unsafe_allow_html=True)
        sel_pu = st.selectbox("Utility", ["Electricity","Water","Fuel","Gas"], key="pc_util")
        pc_top = data_eng.df[
            (data_eng.df["year"] == 2024) & (data_eng.df["utility"] == sel_pu)
        ].nlargest(15, "per_capita")
        fig_pctop = chart_builder.bar_chart(
            pc_top, x="country", y="per_capita",
            title=f"Top 15 Per-Capita {sel_pu} Consumers (2024)"
        )
        fig_pctop.update_layout(height=340)
        st.plotly_chart(fig_pctop, use_container_width=True)

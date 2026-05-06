"""
Chart Builder – Plotly chart factory with UIP dark theme.
All charts share the same visual language: dark bg, cyan/green/amber accents.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

COLORS = {
    "bg_primary":   "#050A14",
    "bg_card":      "#0D1F3C",
    "bg_plot":      "#070F20",
    "grid":         "rgba(0,229,255,0.06)",
    "cyan":         "#00E5FF",
    "green":        "#00FF88",
    "amber":        "#FFB300",
    "red":          "#FF3D71",
    "purple":       "#9C6FDE",
    "text":         "#E8F4FD",
    "text_muted":   "#4A6A8A",
    "Water":        "#00E5FF",
    "Electricity":  "#FFB300",
    "Fuel":         "#FF6B35",
    "Gas":          "#00FF88",
}

PALETTE = ["#00E5FF", "#00FF88", "#FFB300", "#FF6B35", "#9C6FDE", "#FF3D71"]

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["bg_card"],
    plot_bgcolor=COLORS["bg_plot"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,229,255,0.1)",
        borderwidth=1,
        font=dict(size=11, color=COLORS["text_muted"]),
    ),
    xaxis=dict(
        gridcolor=COLORS["grid"], zeroline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
        linecolor="rgba(0,229,255,0.1)",
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"], zeroline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
        linecolor="rgba(0,229,255,0.1)",
    ),
)


class ChartBuilder:

    # ── 1. Forecast Chart ───────────────────────────────────────
    def forecast_chart(self, result: dict) -> go.Figure:
        df = result["df"]
        hist = df[df["type"] == "historical"]
        fut  = df[df["type"] == "forecast"]
        utility = result["utility"]
        color   = COLORS.get(utility, COLORS["cyan"])

        fig = go.Figure()

        # CI ribbon – future
        fig.add_trace(go.Scatter(
            x=list(fut["year"]) + list(fut["year"])[::-1],
            y=list(fut["upper_90"]) + list(fut["lower_90"])[::-1],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI", showlegend=True,
        ))

        # Historical actual
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["actual"],
            mode="lines+markers",
            line=dict(color=COLORS["text_muted"], width=2, dash="dot"),
            marker=dict(size=5, color=COLORS["text_muted"]),
            name="Actual",
        ))

        # Historical predicted
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["predicted"],
            mode="lines",
            line=dict(color=color, width=2),
            name="Model fit",
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fut["year"], y=fut["predicted"],
            mode="lines+markers",
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(size=8, color=color,
                        line=dict(width=2, color=COLORS["bg_card"])),
            name="Forecast",
        ))

        # Vertical separator
        split = hist["year"].max()
        fig.add_vline(x=split + 0.5, line_dash="dash",
                      line_color="rgba(255,255,255,0.15)", line_width=1)
        fig.add_annotation(x=split + 0.5, y=1, yref="paper",
                            text="FORECAST →", showarrow=False,
                            font=dict(size=9, color=COLORS["text_muted"]),
                            bgcolor="rgba(0,0,0,0.5)", borderpad=4, xanchor="left")

        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=f"{result['country']} · {utility} Demand Forecast",
                               font=dict(size=14, color=COLORS["text"]), x=0, xanchor="left")
        layout["yaxis"]["title"] = "Consumption (TWh / Km³ / Bn m³)"
        layout["xaxis"]["title"] = "Year"
        fig.update_layout(**layout)
        return fig

    # ── 2. Time-series Multi-line ───────────────────────────────
    def time_series_multi(self, df: pd.DataFrame,
                          x_col: str, y_col: str,
                          color_col: str, title: str) -> go.Figure:
        fig = go.Figure()
        for i, (name, grp) in enumerate(df.groupby(color_col)):
            color = COLORS.get(name, PALETTE[i % len(PALETTE)])
            fig.add_trace(go.Scatter(
                x=grp[x_col], y=grp[y_col],
                mode="lines", name=name,
                line=dict(color=color, width=2.5),
            ))
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        fig.update_layout(**layout)
        return fig

    # ── 3. Bar Chart ────────────────────────────────────────────
    def bar_chart(self, df: pd.DataFrame, x: str, y: str,
                  color_col: str = None, title: str = "",
                  orientation: str = "v") -> go.Figure:
        fig = go.Figure()
        if color_col and color_col in df.columns:
            for i, (name, grp) in enumerate(df.groupby(color_col)):
                col = COLORS.get(name, PALETTE[i % len(PALETTE)])
                fig.add_trace(go.Bar(
                    x=grp[x] if orientation == "v" else grp[y],
                    y=grp[y] if orientation == "v" else grp[x],
                    name=name, marker_color=col, orientation=orientation,
                ))
        else:
            fig.add_trace(go.Bar(
                x=df[x] if orientation == "v" else df[y],
                y=df[y] if orientation == "v" else df[x],
                marker=dict(
                    color=df[y] if orientation == "v" else df[x],
                    colorscale=[[0,"#0A2040"],[0.5,COLORS["cyan"]],[1,COLORS["green"]]],
                    showscale=False,
                ),
                orientation=orientation,
            ))
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        layout["barmode"] = "group"
        fig.update_layout(**layout)
        return fig

    # ── 4. Heatmap ──────────────────────────────────────────────
    def heatmap(self, df: pd.DataFrame, x: str, y: str,
                z: str, title: str = "") -> go.Figure:
        pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#050A14"], [0.2, "#0A2040"],
                [0.5, COLORS["cyan"]], [0.75, COLORS["amber"]],
                [1.0, COLORS["red"]],
            ],
            showscale=True,
            hovertemplate="%{y} | %{x}: <b>%{z:.1f}</b><extra></extra>",
        ))
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        fig.update_xaxes(tickfont=dict(size=9, color=COLORS["text_muted"]),
                         gridcolor=COLORS["grid"])
        fig.update_yaxes(tickfont=dict(size=9, color=COLORS["text_muted"]),
                         gridcolor=COLORS["grid"])
        return fig

    # ── 5. Scatter / Bubble ─────────────────────────────────────
    def bubble_chart(self, df: pd.DataFrame,
                     x: str, y: str, size: str,
                     color_col: str = None,
                     title: str = "") -> go.Figure:
        fig = go.Figure()
        groups = df.groupby(color_col) if color_col else [(None, df)]
        for i, (name, grp) in enumerate(groups):
            col = COLORS.get(name, PALETTE[i % len(PALETTE)])
            fig.add_trace(go.Scatter(
                x=grp[x], y=grp[y],
                mode="markers",
                name=name or "All",
                marker=dict(
                    size=grp[size],
                    sizemode="area",
                    sizeref=2. * grp[size].max() / (40. ** 2),
                    sizemin=4,
                    color=col,
                    opacity=0.75,
                    line=dict(width=1, color=COLORS["bg_card"]),
                ),
                text=grp.get("country", grp.index),
                hovertemplate="<b>%{text}</b><br>"
                              f"{x}: %{{x:.1f}}<br>"
                              f"{y}: %{{y:.1f}}<extra></extra>",
            ))
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        layout["xaxis"]["title"] = x
        layout["yaxis"]["title"] = y
        fig.update_layout(**layout)
        return fig

    # ── 6. Anomaly Timeline ─────────────────────────────────────
    def anomaly_timeline(self, df: pd.DataFrame, title: str = "") -> go.Figure:
        fig = go.Figure()
        normal  = df[df["anomaly"] == 0]
        anomaly = df[df["anomaly"] == 1]

        fig.add_trace(go.Scatter(
            x=normal["dt"], y=normal["value"],
            mode="lines",
            line=dict(color=COLORS["text_muted"], width=1.5),
            name="Normal",
        ))
        fig.add_trace(go.Scatter(
            x=normal["dt"], y=normal["rolling_mean"],
            mode="lines",
            line=dict(color=COLORS["cyan"], width=1.5, dash="dash"),
            name="Rolling Mean",
        ))
        if not anomaly.empty:
            fig.add_trace(go.Scatter(
                x=anomaly["dt"], y=anomaly["value"],
                mode="markers",
                marker=dict(size=12, color=COLORS["red"],
                            symbol="x-thin-open", line=dict(width=3)),
                name="Anomaly",
            ))

        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        fig.update_layout(**layout)
        return fig

    # ── 7. Gauge Chart ──────────────────────────────────────────
    def gauge(self, value: float, min_val: float, max_val: float,
              label: str, threshold: float = 0.75) -> go.Figure:
        pct  = (value - min_val) / (max_val - min_val)
        col  = COLORS["green"] if pct < 0.5 else \
               COLORS["amber"] if pct < threshold else COLORS["red"]

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={"reference": max_val * 0.6, "valueformat": ".1f",
                   "increasing": {"color": COLORS["red"]},
                   "decreasing": {"color": COLORS["green"]}},
            title={"text": label, "font": {"size": 12, "color": COLORS["text_muted"]}},
            number={"font": {"size": 28, "color": COLORS["text"],
                             "family": "Syne, sans-serif"}},
            gauge={
                "axis": {"range": [min_val, max_val],
                         "tickcolor": COLORS["text_muted"], "dtick": (max_val-min_val)/5},
                "bar": {"color": col, "thickness": 0.25},
                "bgcolor": COLORS["bg_plot"],
                "bordercolor": COLORS["grid"],
                "steps": [
                    {"range": [min_val, max_val * 0.5],  "color": "rgba(0,255,136,0.06)"},
                    {"range": [max_val * 0.5, max_val * 0.75], "color": "rgba(255,179,0,0.06)"},
                    {"range": [max_val * 0.75, max_val],  "color": "rgba(255,61,113,0.06)"},
                ],
                "threshold": {"line": {"color": COLORS["red"], "width": 2},
                              "thickness": 0.75, "value": max_val * threshold},
            },
        ))
        fig.update_layout(
            paper_bgcolor=COLORS["bg_card"],
            plot_bgcolor=COLORS["bg_plot"],
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=40, b=20),
            height=200,
        )
        return fig

    # ── 8. Treemap ──────────────────────────────────────────────
    def treemap(self, df: pd.DataFrame, path_cols: list,
                values: str, title: str = "") -> go.Figure:
        df = df.copy()
        df[values] = df[values].abs()
        fig = px.treemap(df, path=path_cols, values=values,
                         color=values,
                         color_continuous_scale=[
                             [0, "#0A2040"], [0.5, COLORS["cyan"]], [1, COLORS["green"]]
                         ])
        fig.update_traces(
            marker_line_color=COLORS["bg_card"],
            marker_line_width=2,
            textfont=dict(size=11, color="white"),
        )
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        fig.update_layout(**layout)
        return fig

    # ── 9. Feature Importance ───────────────────────────────────
    def feature_importance(self, importance_dict: dict,
                           title: str = "Feature Importance") -> go.Figure:
        items = sorted(importance_dict.items(), key=lambda x: x[1])
        labels = [i[0] for i in items]
        values = [i[1] for i in items]
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, "#0A2040"], [1, COLORS["cyan"]]],
                showscale=False,
            ),
        ))
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        layout["xaxis"]["title"] = "Importance"
        layout["height"] = 300
        fig.update_layout(**layout)
        return fig

    # ── 10. Choropleth-style Scatter Geo ───────────────────────
    def geo_bubble(self, df: pd.DataFrame,
                   size_col: str, color_col: str,
                   title: str = "") -> go.Figure:
        country_coords = {
            "USA": (37.09, -95.71), "China": (35.86, 104.19), "India": (20.59, 78.96),
            "Germany": (51.16, 10.45), "Saudi Arabia": (23.89, 45.08),
            "Canada": (56.13, -106.35), "Mexico": (23.63, -102.55),
            "France": (46.23, 2.21), "UK": (55.38, -3.44), "Italy": (41.87, 12.57),
            "Spain": (40.46, -3.75), "Poland": (51.92, 19.14), "Japan": (36.20, 138.25),
            "South Korea": (35.91, 127.77), "Australia": (-25.27, 133.78),
            "UAE": (23.42, 53.85), "Iran": (32.43, 53.69), "Iraq": (33.22, 43.68),
            "Egypt": (26.82, 30.80), "Nigeria": (9.08, 8.68),
            "South Africa": (-30.56, 22.94), "Ethiopia": (9.15, 40.49),
        }
        df = df.copy()
        df["lat"] = df["country"].map(lambda c: country_coords.get(c, (0,0))[0])
        df["lon"] = df["country"].map(lambda c: country_coords.get(c, (0,0))[1])

        fig = go.Figure(go.Scattergeo(
            lat=df["lat"], lon=df["lon"],
            text=df["country"],
            marker=dict(
                size=df[size_col] / df[size_col].max() * 50 + 8,
                color=df[color_col],
                colorscale=[[0,"#050A14"],[0.4,COLORS["cyan"]],[0.7,COLORS["amber"]],[1,COLORS["red"]]],
                showscale=True,
                colorbar=dict(
                    bgcolor=COLORS["bg_card"],
                    tickfont=dict(color=COLORS["text_muted"], size=9),
                    title=dict(text=color_col, font=dict(color=COLORS["text_muted"], size=9)),
                ),
                opacity=0.8,
                line=dict(width=1.5, color=COLORS["bg_card"]),
            ),
            hovertemplate="<b>%{text}</b><br>"
                          f"Consumption: %{{marker.size:.0f}}<br>"
                          f"Loss: %{{marker.color:.1f}}%<extra></extra>",
        ))
        fig.update_geos(
            showland=True, landcolor="#0A1628",
            showocean=True, oceancolor="#060D1E",
            showcoastlines=True, coastlinecolor="rgba(0,229,255,0.15)",
            showframe=False, projection_type="natural earth",
            bgcolor=COLORS["bg_card"],
            showlakes=False,
            showcountries=True, countrycolor="rgba(0,229,255,0.08)",
        )
        layout = {**BASE_LAYOUT}
        layout["title"] = dict(text=title, font=dict(size=13, color=COLORS["text"]),
                               x=0, xanchor="left")
        layout["height"] = 450
        layout["geo"] = dict(bgcolor=COLORS["bg_card"])
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        return fig

    # ── 11. Donut ───────────────────────────────────────────────
    def donut(self, labels: list, values: list, title: str = "") -> go.Figure:
        colors = [COLORS.get(l, PALETTE[i % len(PALETTE)]) for i, l in enumerate(labels)]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors,
                        line=dict(color=COLORS["bg_card"], width=3)),
            textfont=dict(size=11, color="white"),
        ))
        fig.add_annotation(text=title, x=0.5, y=0.5, font_size=11,
                           showarrow=False, font_color=COLORS["text_muted"])
        layout = {**BASE_LAYOUT}
        layout["showlegend"] = True
        layout["margin"] = dict(l=10, r=10, t=30, b=10)
        fig.update_layout(**layout)
        return fig

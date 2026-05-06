"""
Data Engine – Realistic synthetic dataset mirroring Kaggle's
"Global Energy & Water Consumption" dataset structure.

Columns align with: https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption
and supplemented with water/fuel data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

REGIONS = {
    "North America": ["USA", "Canada", "Mexico"],
    "Europe":        ["Germany", "France", "UK", "Italy", "Spain", "Poland"],
    "Asia Pacific":  ["China", "India", "Japan", "South Korea", "Australia"],
    "Middle East":   ["Saudi Arabia", "UAE", "Iran", "Iraq"],
    "Africa":        ["Egypt", "Nigeria", "South Africa", "Ethiopia"],
}

UTILITY_COLORS = {
    "Water":       "#00E5FF",
    "Electricity": "#FFB300",
    "Fuel":        "#FF6B35",
    "Gas":         "#00FF88",
}

# Seasonal multipliers (monthly) – captures demand seasonality
SEASONAL = {
    "Water":       [0.85, 0.82, 0.90, 0.95, 1.05, 1.15, 1.20, 1.18, 1.05, 0.98, 0.88, 0.80],
    "Electricity": [1.15, 1.10, 0.95, 0.88, 0.90, 1.10, 1.25, 1.22, 0.95, 0.85, 1.00, 1.18],
    "Fuel":        [1.05, 1.02, 1.00, 1.02, 1.05, 1.08, 1.10, 1.08, 1.00, 0.98, 0.97, 1.05],
    "Gas":         [1.30, 1.25, 1.05, 0.85, 0.70, 0.60, 0.58, 0.62, 0.78, 1.00, 1.20, 1.35],
}

GROWTH_RATES = {
    "Water":       0.018,
    "Electricity": 0.028,
    "Fuel":        0.012,
    "Gas":         0.022,
}

BASE_CONSUMPTION = {   # TWh or Billion Cubic Metres or Km³ per country-year (normalised)
    "Water":       {"USA": 480, "China": 610, "India": 760, "Germany": 180, "Saudi Arabia": 220,
                    "Canada": 250, "Mexico": 190, "France": 165, "UK": 150, "Italy": 155,
                    "Spain": 135, "Poland": 120, "Japan": 340, "South Korea": 180, "Australia": 145,
                    "UAE": 130, "Iran": 210, "Iraq": 200, "Egypt": 230, "Nigeria": 175,
                    "South Africa": 145, "Ethiopia": 160},
    "Electricity": {"USA": 4200, "China": 8800, "India": 1700, "Germany": 510, "Saudi Arabia": 320,
                    "Canada": 620, "Mexico": 340, "France": 480, "UK": 400, "Italy": 370,
                    "Spain": 280, "Poland": 190, "Japan": 1000, "South Korea": 580, "Australia": 260,
                    "UAE": 140, "Iran": 280, "Iraq": 90, "Egypt": 185, "Nigeria": 30,
                    "South Africa": 230, "Ethiopia": 12},
    "Fuel":        {"USA": 820, "China": 680, "India": 240, "Germany": 240, "Saudi Arabia": 170,
                    "Canada": 280, "Mexico": 145, "France": 195, "UK": 180, "Italy": 170,
                    "Spain": 130, "Poland": 90, "Japan": 190, "South Korea": 130, "Australia": 68,
                    "UAE": 95, "Iran": 130, "Iraq": 45, "Egypt": 82, "Nigeria": 35,
                    "South Africa": 42, "Ethiopia": 8},
    "Gas":         {"USA": 900, "China": 380, "India": 65, "Germany": 95, "Saudi Arabia": 120,
                    "Canada": 130, "Mexico": 90, "France": 45, "UK": 80, "Italy": 75,
                    "Spain": 35, "Poland": 22, "Japan": 115, "South Korea": 55, "Australia": 42,
                    "UAE": 80, "Iran": 220, "Iraq": 30, "Egypt": 58, "Nigeria": 25,
                    "South Africa": 5, "Ethiopia": 0.5},
}

class DataEngine:
    def __init__(self):
        np.random.seed(42)
        self.df = self._build_dataset()
        self.df_monthly = self._build_monthly_series()
        self.countries = sorted(self.df["country"].unique().tolist())
        self.regions_list = list(REGIONS.keys())
        self.utility_colors = UTILITY_COLORS

    # ── Main historical dataset ────────────────────────────────
    def _build_dataset(self):
        records = []
        all_countries = [c for cs in REGIONS.values() for c in cs]

        for year in range(2000, 2025):
            for country, region_name in [(c, r) for r, cs in REGIONS.items() for c in cs]:
                for utility in ["Water", "Electricity", "Fuel", "Gas"]:
                    base = BASE_CONSUMPTION[utility].get(country, 50)
                    growth = (1 + GROWTH_RATES[utility]) ** (year - 2000)
                    # Russia-Ukraine shock (2022+) – supply disruptions
                    shock = 1.0
                    if utility in ["Fuel", "Gas"] and year >= 2022:
                        shock = np.random.uniform(0.80, 0.95)
                    if utility == "Gas" and country in ["Germany", "Poland", "France"] and year >= 2022:
                        shock = np.random.uniform(0.70, 0.85)
                    # Covid dip
                    if year == 2020:
                        shock *= np.random.uniform(0.88, 0.96)

                    consumption = base * growth * shock * np.random.uniform(0.95, 1.05)
                    supply = consumption * np.random.uniform(0.90, 1.08)
                    gap = supply - consumption
                    loss_pct = np.random.uniform(0.05, 0.20)
                    loss = consumption * loss_pct
                    price = self._price(utility, year, country, shock)
                    co2 = self._co2(utility, consumption, country)

                    records.append({
                        "year": year,
                        "country": country,
                        "region": region_name,
                        "utility": utility,
                        "consumption": round(consumption, 2),
                        "supply": round(supply, 2),
                        "gap": round(gap, 2),
                        "loss": round(loss, 2),
                        "loss_pct": round(loss_pct * 100, 1),
                        "price": round(price, 3),
                        "co2": round(co2, 2),
                        "pop_million": self._pop(country, year),
                        "gdp_bn": self._gdp(country, year),
                    })

        df = pd.DataFrame(records)
        df["per_capita"] = (df["consumption"] / df["pop_million"]).round(3)
        df["intensity"] = (df["consumption"] / df["gdp_bn"]).round(4)
        df["shortage_risk"] = (df["gap"] < 0).astype(int)
        return df

    # ── Monthly time-series ────────────────────────────────────
    def _build_monthly_series(self):
        records = []
        base_date = datetime(2019, 1, 1)
        countries_sample = ["USA", "China", "Germany", "India", "Saudi Arabia", "Nigeria"]

        for country in countries_sample:
            region = next(r for r, cs in REGIONS.items() if country in cs)
            for utility in ["Water", "Electricity", "Fuel", "Gas"]:
                base = BASE_CONSUMPTION[utility].get(country, 50)
                for m in range(73):  # 2019-01 → 2025-01
                    dt = base_date + timedelta(days=30 * m)
                    yr = dt.year
                    mo = dt.month - 1
                    growth = (1 + GROWTH_RATES[utility]) ** (yr - 2019)
                    seasonal = SEASONAL[utility][mo]
                    shock = 1.0
                    if utility in ["Fuel", "Gas"] and yr >= 2022:
                        shock = 0.88 if country in ["Germany"] else 0.95
                    if yr == 2020 and 3 <= dt.month <= 8:
                        shock *= 0.90
                    noise = np.random.uniform(0.97, 1.03)
                    val = base / 12 * growth * seasonal * shock * noise
                    records.append({
                        "date": dt.strftime("%Y-%m"),
                        "dt": dt,
                        "country": country,
                        "region": region,
                        "utility": utility,
                        "value": round(val, 2),
                        "month": dt.month,
                        "year": dt.year,
                    })

        return pd.DataFrame(records).sort_values("dt").reset_index(drop=True)

    # ── Helper generators ──────────────────────────────────────
    def _price(self, utility, year, country, shock):
        base = {"Water": 0.8, "Electricity": 0.12, "Fuel": 1.5, "Gas": 6.5}[utility]
        inflation = (1.025) ** (year - 2000)
        premium = 1.3 if country in ["Germany", "UK", "Japan", "Australia"] else 1.0
        return base * inflation * premium * shock * np.random.uniform(0.95, 1.05)

    def _co2(self, utility, consumption, country):
        factor = {"Water": 0.05, "Electricity": 0.45, "Fuel": 2.5, "Gas": 1.8}
        country_factor = 1.4 if country in ["China", "India", "South Africa"] else \
                         0.7 if country in ["France", "Norway", "Canada"] else 1.0
        return consumption * factor.get(utility, 1) * country_factor * np.random.uniform(0.95, 1.05)

    def _pop(self, country, year):
        base = {"USA": 330, "China": 1400, "India": 1380, "Germany": 84, "Saudi Arabia": 35,
                "Canada": 38, "Mexico": 130, "France": 68, "UK": 67, "Italy": 60,
                "Spain": 47, "Poland": 38, "Japan": 126, "South Korea": 52, "Australia": 26,
                "UAE": 10, "Iran": 85, "Iraq": 41, "Egypt": 104, "Nigeria": 220,
                "South Africa": 60, "Ethiopia": 120}
        return round(base.get(country, 50) * (1.01) ** (year - 2020), 1)

    def _gdp(self, country, year):
        base = {"USA": 23000, "China": 17700, "India": 3100, "Germany": 4200, "Saudi Arabia": 800,
                "Canada": 2000, "Mexico": 1300, "France": 2900, "UK": 3100, "Italy": 2100,
                "Spain": 1500, "Poland": 700, "Japan": 4900, "South Korea": 1800, "Australia": 1700,
                "UAE": 420, "Iran": 260, "Iraq": 210, "Egypt": 420, "Nigeria": 440,
                "South Africa": 420, "Ethiopia": 120}
        return round(base.get(country, 200) * (1.025) ** (year - 2022), 0)

    # ── Public API ─────────────────────────────────────────────
    def get_annual(self, utility="All", region="Global", year_range=(2015, 2024)):
        df = self.df.copy()
        if utility != "All":
            df = df[df["utility"] == utility]
        if region != "Global":
            df = df[df["region"] == region]
        return df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    def get_monthly(self, country=None, utility=None):
        df = self.df_monthly.copy()
        if country:
            df = df[df["country"] == country]
        if utility:
            df = df[df["utility"] == utility]
        return df

    def get_kpis(self):
        latest = self.df[self.df["year"] == 2024]
        prev   = self.df[self.df["year"] == 2023]
        total  = latest["consumption"].sum()
        prev_t = prev["consumption"].sum()
        shortage = (latest["gap"] < 0).mean() * 100
        loss     = latest["loss_pct"].mean()
        co2      = latest["co2"].sum()
        return {
            "total_consumption": total,
            "consumption_delta": (total - prev_t) / prev_t * 100,
            "shortage_risk_pct": shortage,
            "avg_loss_pct": loss,
            "co2_mn_tonnes": co2 / 1000,
            "countries": latest["country"].nunique(),
        }

    def get_country_summary(self, year=2024):
        df = self.df[self.df["year"] == year].groupby("country").agg(
            total_consumption=("consumption", "sum"),
            avg_loss_pct=("loss_pct", "mean"),
            shortage_events=("shortage_risk", "sum"),
            co2=("co2", "sum"),
            region=("region", "first"),
        ).reset_index()
        return df

    def get_alerts(self):
        df = self.df[(self.df["year"] == 2024) & (self.df["gap"] < 0)]
        df = df.nlargest(8, "loss_pct")
        alerts = []
        for _, row in df.iterrows():
            severity = "critical" if row["loss_pct"] > 15 else "warning"
            alerts.append({
                "country": row["country"],
                "utility": row["utility"],
                "loss_pct": row["loss_pct"],
                "severity": severity,
                "message": f"{row['utility']} deficit detected — {row['loss_pct']:.1f}% distribution loss",
            })
        return alerts

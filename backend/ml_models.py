"""
ML Models – Forecasting & Anomaly Detection
==========================================
· ForecastEngine   → Random Forest + Gradient Boosting ensemble
· AnomalyDetector  → Isolation Forest + statistical thresholding
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  Forecast Engine
# ══════════════════════════════════════════════════════════════
class ForecastEngine:
    """
    Trains separate models per (country, utility) pair.
    Features: year, gdp, population, price, lagged consumption,
              rolling-mean, seasonality encodings.
    Uses RF + GBM stacked ensemble.
    """

    def __init__(self, data_engine):
        self.data = data_engine
        self._models: dict = {}
        self._scalers: dict = {}
        self._metrics: dict = {}
        self._feature_importance: dict = {}
        self._train_all()

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("year").copy()
        df["lag1"]     = df["consumption"].shift(1)
        df["lag2"]     = df["consumption"].shift(2)
        df["lag3"]     = df["consumption"].shift(3)
        df["roll3"]    = df["consumption"].shift(1).rolling(3).mean()
        df["roll5"]    = df["consumption"].shift(1).rolling(5).mean()
        df["yoy"]      = df["consumption"].pct_change()
        df["year_sq"]  = df["year"] ** 2
        df["log_gdp"]  = np.log1p(df["gdp_bn"])
        df["log_pop"]  = np.log1p(df["pop_million"])
        return df.dropna()

    def _train_all(self):
        df = self.data.df.copy()
        combos = df.groupby(["country", "utility"])

        for (country, utility), grp in combos:
            grp = self._feature_engineer(grp)
            if len(grp) < 8:
                continue

            features = ["year", "year_sq", "lag1", "lag2", "lag3",
                        "roll3", "roll5", "yoy", "price",
                        "log_gdp", "log_pop", "loss_pct"]
            X = grp[features].values
            y = grp["consumption"].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            rf  = RandomForestRegressor(n_estimators=120, max_depth=8,
                                        min_samples_leaf=2, random_state=42, n_jobs=-1)
            gbm = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                            learning_rate=0.08, random_state=42)
            rf.fit(X_tr_s, y_tr)
            gbm.fit(X_tr_s, y_tr)

            # Ensemble: 60% RF + 40% GBM
            y_pred = 0.6 * rf.predict(X_te_s) + 0.4 * gbm.predict(X_te_s)
            mape = mean_absolute_percentage_error(y_te, y_pred) * 100
            r2   = r2_score(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))

            key = (country, utility)
            self._models[key]  = (rf, gbm, scaler, features, grp)
            self._metrics[key] = {"mape": round(mape, 2), "r2": round(r2, 3),
                                   "rmse": round(rmse, 2)}
            self._feature_importance[key] = dict(zip(features, rf.feature_importances_))

    def predict(self, country: str, utility: str,
                forecast_years: int = 5) -> dict:
        """Return historical fit + future forecast DataFrame."""
        key = (country, utility)
        if key not in self._models:
            return None

        rf, gbm, scaler, features, hist = self._models[key]

        # ── Historical predictions ───────────────────────────
        hist_fe = self._feature_engineer(hist.copy())
        X_hist  = scaler.transform(hist_fe[features].values)
        y_hist_pred = 0.6 * rf.predict(X_hist) + 0.4 * gbm.predict(X_hist)

        # ── Future forecast ──────────────────────────────────
        last_year = int(hist["year"].max())
        last_row  = hist_fe.iloc[-1].to_dict()

        future_rows = []
        consumption_hist = list(hist_fe["consumption"].values)

        for i in range(1, forecast_years + 1):
            yr = last_year + i
            c1 = consumption_hist[-1]
            c2 = consumption_hist[-2]
            c3 = consumption_hist[-3]
            roll3 = np.mean(consumption_hist[-3:])
            roll5 = np.mean(consumption_hist[-5:]) if len(consumption_hist) >= 5 else roll3
            yoy   = (c1 - c2) / (c2 + 1e-8)
            gdp_projected = last_row["log_gdp"] + np.log(1.025) * i
            pop_projected = last_row["log_pop"] + np.log(1.01) * i

            row = {
                "year": yr, "year_sq": yr ** 2,
                "lag1": c1, "lag2": c2, "lag3": c3,
                "roll3": roll3, "roll5": roll5, "yoy": yoy,
                "price": last_row["price"] * (1.02 ** i),
                "log_gdp": gdp_projected, "log_pop": pop_projected,
                "loss_pct": last_row["loss_pct"],
            }
            X_row = scaler.transform([[row[f] for f in features]])
            pred  = 0.6 * rf.predict(X_row)[0] + 0.4 * gbm.predict(X_row)[0]

            # Confidence interval (±1 RMSE scaled)
            rmse  = self._metrics[key]["rmse"]
            lower = pred - 1.65 * rmse
            upper = pred + 1.65 * rmse

            future_rows.append({
                "year": yr, "predicted": round(pred, 2),
                "lower_90": round(max(0, lower), 2),
                "upper_90": round(upper, 2),
                "type": "forecast",
            })
            consumption_hist.append(pred)

        # ── Assemble result ──────────────────────────────────
        hist_df = pd.DataFrame({
            "year": hist_fe["year"].values,
            "actual": hist_fe["consumption"].values,
            "predicted": y_hist_pred,
            "lower_90": y_hist_pred * 0.95,
            "upper_90": y_hist_pred * 1.05,
            "type": "historical",
        })
        fut_df   = pd.DataFrame(future_rows)
        combined = pd.concat([hist_df, fut_df], ignore_index=True)

        return {
            "df": combined,
            "metrics": self._metrics[key],
            "feature_importance": self._feature_importance[key],
            "country": country,
            "utility": utility,
        }

    def bulk_forecast(self, utility: str, forecast_years: int = 3) -> pd.DataFrame:
        """Forecast for all countries for a given utility."""
        rows = []
        for (country, util), _ in self._models.items():
            if util != utility:
                continue
            result = self.predict(country, utility, forecast_years)
            if result is None:
                continue
            fut = result["df"][result["df"]["type"] == "forecast"]
            for _, r in fut.iterrows():
                rows.append({
                    "country": country, "year": r["year"],
                    "predicted": r["predicted"],
                    "utility": utility,
                })
        return pd.DataFrame(rows)

    def get_model_summary(self) -> pd.DataFrame:
        rows = []
        for (country, utility), m in self._metrics.items():
            rows.append({
                "Country": country, "Utility": utility,
                "MAPE (%)": m["mape"], "R²": m["r2"], "RMSE": m["rmse"],
            })
        return pd.DataFrame(rows).sort_values("MAPE (%)")


# ══════════════════════════════════════════════════════════════
#  Anomaly Detector
# ══════════════════════════════════════════════════════════════
class AnomalyDetector:
    """
    Isolation Forest for detecting unusual consumption/loss events.
    Also computes statistical z-score anomalies.
    """

    def __init__(self, data_engine):
        self.data = data_engine
        self._models: dict = {}
        self._train_all()

    def _train_all(self):
        df = self.data.df.copy()
        for utility, grp in df.groupby("utility"):
            features = ["consumption", "loss_pct", "gap", "price", "per_capita"]
            X = grp[features].dropna().values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            iso = IsolationForest(n_estimators=100, contamination=0.08,
                                  random_state=42, n_jobs=-1)
            iso.fit(X_s)
            self._models[utility] = (iso, scaler, features)

    def detect(self, utility: str = "Electricity", year: int = 2024) -> pd.DataFrame:
        if utility not in self._models:
            return pd.DataFrame()

        iso, scaler, features = self._models[utility]
        df = self.data.df[(self.data.df["utility"] == utility) &
                          (self.data.df["year"] == year)].copy()
        df = df.dropna(subset=features)

        X_s = scaler.transform(df[features].values)
        labels  = iso.predict(X_s)           # -1 = anomaly, 1 = normal
        scores  = iso.score_samples(X_s)     # lower = more anomalous

        df["anomaly"]       = (labels == -1).astype(int)
        df["anomaly_score"] = -scores        # flip so higher = more anomalous
        df["z_consumption"] = (
            (df["consumption"] - df["consumption"].mean()) / df["consumption"].std()
        ).round(2)
        df["severity"] = pd.cut(
            df["anomaly_score"],
            bins=[0, 0.1, 0.2, np.inf],
            labels=["Low", "Medium", "High"]
        )
        return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    def time_series_anomaly(self, country: str, utility: str) -> pd.DataFrame:
        """Monthly anomaly detection on time-series."""
        df = self.data.get_monthly(country=country, utility=utility).copy()
        if df.empty:
            return df

        # Rolling z-score
        df = df.sort_values("dt")
        df["rolling_mean"] = df["value"].rolling(6, min_periods=1).mean()
        df["rolling_std"]  = df["value"].rolling(6, min_periods=1).std().fillna(1)
        df["z_score"]      = ((df["value"] - df["rolling_mean"]) / df["rolling_std"]).round(2)
        df["anomaly"]      = (df["z_score"].abs() > 2.0).astype(int)
        df["direction"]    = np.where(df["z_score"] > 0, "spike", "dip")
        return df

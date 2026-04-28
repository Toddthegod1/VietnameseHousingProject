"""
Predict 30-day housing price growth by district.

This is a harder and more useful target than next-day price level because the
model must estimate appreciation, not simply repeat the most recent price.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "HousePricingHCM.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 30
FEATURE_COLS = [
    "District", "Price", "year", "month", "quarter", "dayofyear",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "roll_mean_3", "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_3", "roll_std_7", "roll_std_14", "roll_std_30",
    "growth_7d", "growth_14d", "growth_30d",
]
NUMERIC_COLS = [c for c in FEATURE_COLS if c != "District"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_long() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    long = df.melt(id_vars="Date", var_name="District", value_name="Price")
    long["Date"] = pd.to_datetime(long["Date"])
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    return long.dropna(subset=["Price"]).sort_values(["District", "Date"]).reset_index(drop=True)


def build_growth_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["dayofyear"] = df["Date"].dt.dayofyear

    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby("District")["Price"].shift(lag)

    for window in [3, 7, 14, 30]:
        grp = df.groupby("District")["Price"]
        df[f"roll_mean_{window}"] = grp.transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"roll_std_{window}"] = grp.transform(lambda x: x.shift(1).rolling(window).std())

    for days in [7, 14, 30]:
        df[f"growth_{days}d"] = (df["Price"] / df.groupby("District")["Price"].shift(days) - 1) * 100

    df["future_price_30d"] = df.groupby("District")["Price"].shift(-HORIZON)
    df["target_growth_30d"] = (df["future_price_30d"] / df["Price"] - 1) * 100

    return df.dropna().reset_index(drop=True)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_date = df["Date"].quantile(0.8)
    return df[df["Date"] <= split_date].copy(), df[df["Date"] > split_date].copy()


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", make_one_hot_encoder(), ["District"]),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
        verbosity=0,
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


def train_and_predict(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    train, test = time_split(df)
    pipe = build_pipeline()
    pipe.fit(train[FEATURE_COLS], train["target_growth_30d"])

    pred_df = test[["Date", "District", "Price", "future_price_30d", "target_growth_30d"]].copy()
    pred_df["predicted_growth_30d"] = pipe.predict(test[FEATURE_COLS])
    pred_df["absolute_error"] = (pred_df["target_growth_30d"] - pred_df["predicted_growth_30d"]).abs()
    pred_df.to_csv(TABLES_DIR / "growth_30d_predictions.csv", index=False)

    district_metrics = (
        pred_df.groupby("District")
        .apply(
            lambda g: pd.Series(
                {
                    "MAE_pct_points": mean_absolute_error(g["target_growth_30d"], g["predicted_growth_30d"]),
                    "RMSE_pct_points": np.sqrt(mean_squared_error(g["target_growth_30d"], g["predicted_growth_30d"])),
                    "R2": r2_score(g["target_growth_30d"], g["predicted_growth_30d"]),
                    "ActualAvgGrowth30d": g["target_growth_30d"].mean(),
                    "PredictedAvgGrowth30d": g["predicted_growth_30d"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values("PredictedAvgGrowth30d", ascending=False)
    )
    district_metrics.to_csv(TABLES_DIR / "growth_30d_district_metrics.csv", index=False)

    return pred_df, district_metrics, pipe


def predict_latest_by_district(df: pd.DataFrame, pipe: Pipeline) -> pd.DataFrame:
    latest_rows = df.sort_values("Date").groupby("District").tail(1).copy()
    latest_rows["predicted_growth_30d"] = pipe.predict(latest_rows[FEATURE_COLS])
    latest_rows["predicted_price_30d"] = latest_rows["Price"] * (1 + latest_rows["predicted_growth_30d"] / 100)

    latest_forecast = latest_rows[
        ["Date", "District", "Price", "predicted_growth_30d", "predicted_price_30d"]
    ].sort_values("predicted_growth_30d", ascending=False)
    latest_forecast.to_csv(TABLES_DIR / "growth_30d_latest_district_forecast.csv", index=False)
    return latest_forecast


def plot_actual_vs_predicted(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pred_df["target_growth_30d"], pred_df["predicted_growth_30d"], alpha=0.45, s=22)

    low = min(pred_df["target_growth_30d"].min(), pred_df["predicted_growth_30d"].min())
    high = max(pred_df["target_growth_30d"].max(), pred_df["predicted_growth_30d"].max())
    ax.plot([low, high], [low, high], color="black", linestyle="--", lw=1.2, label="Perfect prediction")

    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Actual 30-Day Growth (%)")
    ax.set_ylabel("Predicted 30-Day Growth (%)")
    ax.set_title("Actual vs Predicted 30-Day Price Growth", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "growth_30d_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved growth_30d_actual_vs_predicted.png")


def plot_district_accuracy(district_metrics: pd.DataFrame) -> None:
    ordered = district_metrics.sort_values("MAE_pct_points", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ordered["District"], ordered["MAE_pct_points"], color="#4C78A8", alpha=0.85)
    ax.set_xlabel("MAE (percentage points)")
    ax.set_title("30-Day Growth Forecast Error by District", fontweight="bold")

    for bar in bars:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "growth_30d_error_by_district.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved growth_30d_error_by_district.png")


def plot_latest_growth_forecast(latest_forecast: pd.DataFrame) -> None:
    colors = ["#2E7D32" if value >= 0 else "#C62828" for value in latest_forecast["predicted_growth_30d"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(latest_forecast["District"], latest_forecast["predicted_growth_30d"], color=colors, alpha=0.88)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Predicted 30-Day Growth (%)")
    ax.set_title("Predicted 30-Day Price Growth by District\n(latest available date)", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    for bar in bars:
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.05 if y >= 0 else -0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{y:.2f}%", ha="center", va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_30d_growth_by_district.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved predicted_30d_growth_by_district.png")


def main() -> None:
    df = build_growth_dataset(load_long())
    pred_df, district_metrics, pipe = train_and_predict(df)
    latest_forecast = predict_latest_by_district(df, pipe)

    benchmark_rows = []
    benchmarks = {
        "XGBoost": pred_df["predicted_growth_30d"],
        "NaiveZeroGrowth": np.zeros(len(pred_df)),
        "NaivePersistPast30dGrowth": df.loc[pred_df.index, "growth_30d"].to_numpy(),
    }
    for name, preds in benchmarks.items():
        benchmark_rows.append(
            {
                "Model": name,
                "Horizon": HORIZON,
                "MAE_pct_points": mean_absolute_error(pred_df["target_growth_30d"], preds),
                "RMSE_pct_points": np.sqrt(mean_squared_error(pred_df["target_growth_30d"], preds)),
                "R2": r2_score(pred_df["target_growth_30d"], preds),
            }
        )
    overall_metrics = pd.DataFrame(benchmark_rows).sort_values("MAE_pct_points")
    overall_metrics.to_csv(TABLES_DIR / "growth_30d_overall_metrics.csv", index=False)

    plot_actual_vs_predicted(pred_df)
    plot_district_accuracy(district_metrics)
    plot_latest_growth_forecast(latest_forecast)

    model_row = overall_metrics[overall_metrics["Model"] == "XGBoost"].iloc[0]
    print("\n30-Day Growth Forecast Summary")
    print(overall_metrics.to_string(index=False))
    print(
        "\nXGBoost MAE: "
        f"{model_row['MAE_pct_points']:.3f} percentage points "
        "(lower is better for growth-rate prediction)"
    )
    print("\nLatest district forecast ranking")
    print(latest_forecast.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Train XGBoost models for 1-day, 7-day, and 30-day forecast horizons and compare
accuracy across districts.
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

HORIZONS = [1, 7, 30]
FEATURE_COLS = [
    "District", "year", "month", "quarter", "dayofyear",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "roll_mean_3", "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_3", "roll_std_7", "roll_std_14", "roll_std_30",
]
NUMERIC_COLS = [c for c in FEATURE_COLS if c != "District"]


def load_long() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    long = df.melt(id_vars="Date", var_name="District", value_name="Price")
    long["Date"] = pd.to_datetime(long["Date"])
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    return long.dropna(subset=["Price"]).sort_values(["District", "Date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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

    df["target"] = df.groupby("District")["Price"].shift(-horizon)
    return df.dropna().reset_index(drop=True)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_date = df["Date"].quantile(0.8)
    return df[df["Date"] <= split_date].copy(), df[df["Date"] > split_date].copy()


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["District"]),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        verbosity=0,
    )
    return Pipeline([("pre", preprocessor), ("model", model)])


def evaluate_horizon(horizon: int, long_df: pd.DataFrame) -> dict:
    df = build_features(long_df, horizon)
    train, test = time_split(df)

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    test = test.copy()
    test["predicted"] = preds
    district_rows = []
    for district, grp in test.groupby("District"):
        district_rows.append(
            {
                "horizon": horizon,
                "District": district,
                "MAE": mean_absolute_error(grp["target"], grp["predicted"]),
                "RMSE": np.sqrt(mean_squared_error(grp["target"], grp["predicted"])),
                "R2": r2_score(grp["target"], grp["predicted"]),
            }
        )

    overall = {
        "horizon": horizon,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

    print(
        f"  Horizon {horizon:>2}d - MAE: {overall['MAE']:.3f} | "
        f"RMSE: {overall['RMSE']:.3f} | R2: {overall['R2']:.6f}"
    )
    return {
        "overall": overall,
        "district": district_rows,
        "predictions": test[["Date", "District", "target", "predicted"]],
    }


def plot_horizon_comparison(overall_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#2196F3", "#FF9800", "#F44336"]
    metrics = ["MAE", "RMSE", "R2"]
    labels = ["MAE (Million VND)", "RMSE (Million VND)", "R2 Score"]

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        bars = ax.bar(
            [f"{h}-day" for h in overall_df["horizon"]],
            overall_df[metric],
            color=color,
            alpha=0.85,
            width=0.5,
        )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Forecast Horizon")
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.01,
                f"{value:.4f}" if metric == "R2" else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle("Forecast Accuracy Degrades with Horizon\nXGBoost - HCM Housing Prices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multi_horizon_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved multi_horizon_accuracy.png")


def plot_district_horizon_heatmap(district_df: pd.DataFrame) -> None:
    pivot_mae = district_df.pivot(index="District", columns="horizon", values="MAE")
    pivot_r2 = district_df.pivot(index="District", columns="horizon", values="R2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (axes[0], pivot_mae, "MAE by District & Horizon (Million VND)", ".1f", "RdYlGn_r"),
        (axes[1], pivot_r2, "R2 by District & Horizon", ".3f", "RdYlGn"),
    ]

    for ax, pivot, title, fmt, cmap in panels:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{h}-day" for h in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:{fmt}}", ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_horizon_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved district_horizon_heatmap.png")


def plot_horizon_predictions(preds_by_horizon: dict[int, pd.DataFrame]) -> None:
    district = "District 7"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ["#2196F3", "#FF9800", "#F44336"]

    for ax, (horizon, pred_df), color in zip(axes, preds_by_horizon.items(), colors):
        sub = pred_df[pred_df["District"] == district].sort_values("Date").tail(200)
        ax.plot(sub["Date"], sub["target"], label="Actual", color="black", lw=1.5)
        ax.plot(sub["Date"], sub["predicted"], label="Predicted", color=color, lw=1.5, linestyle="--")
        mae = mean_absolute_error(sub["target"], sub["predicted"])
        ax.set_title(f"{horizon}-Day Ahead Forecast\nMAE: {mae:.2f}", fontweight="bold")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Price (Million VND)")
    fig.suptitle(f"Actual vs Predicted - {district} at Different Forecast Horizons", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_predictions_district7.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved horizon_predictions_district7.png")


def main() -> None:
    long_df = load_long()
    overall_rows = []
    district_rows = []
    preds_by_horizon = {}

    print("Training XGBoost for each forecast horizon...")
    for horizon in HORIZONS:
        result = evaluate_horizon(horizon, long_df)
        overall_rows.append(result["overall"])
        district_rows.extend(result["district"])
        preds_by_horizon[horizon] = result["predictions"]

    overall_df = pd.DataFrame(overall_rows)
    district_df = pd.DataFrame(district_rows)

    overall_df.to_csv(TABLES_DIR / "multi_horizon_overall.csv", index=False)
    district_df.to_csv(TABLES_DIR / "multi_horizon_district.csv", index=False)

    plot_horizon_comparison(overall_df)
    plot_district_horizon_heatmap(district_df)
    plot_horizon_predictions(preds_by_horizon)

    print("\nMulti-Horizon Summary")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()

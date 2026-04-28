"""
Advanced analyses for the housing price pipeline:

1. Feature importance for XGBoost
2. Residual analysis from saved Linear Regression predictions
3. K-means elbow and silhouette diagnostics
4. Price gap convergence/divergence over time
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_modeling_data() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "modeling_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_date = df["Date"].quantile(0.8)
    return df[df["Date"] <= split_date].copy(), df[df["Date"] > split_date].copy()


def build_xgb_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric = [c for c in X_train.columns if c != "District"]
    pre = ColumnTransformer(
        [
            ("cat", make_one_hot_encoder(), ["District"]),
            ("num", StandardScaler(), numeric),
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
    return Pipeline([("pre", pre), ("model", model)])


def feature_importance_analysis() -> None:
    print("Feature Importance")
    df = load_modeling_data()
    train, _ = time_split(df)

    X_train = train.drop(columns=["Date", "Price"])
    y_train = train["Price"]

    pipe = build_xgb_pipeline(X_train)
    pipe.fit(X_train, y_train)

    ohe_cats = pipe["pre"].named_transformers_["cat"].get_feature_names_out(["District"])
    num_cols = [c for c in X_train.columns if c != "District"]
    all_feats = np.concatenate([ohe_cats, num_cols])
    importances = pipe["model"].feature_importances_

    imp_df = (
        pd.DataFrame({"feature": all_feats, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(TABLES_DIR / "feature_importances.csv", index=False)

    def family(name: str) -> str:
        if name.startswith("District_"):
            return "District (OHE)"
        if name.startswith("lag_"):
            return "Lag features"
        if name.startswith("roll_mean_"):
            return "Rolling mean"
        if name.startswith("roll_std_"):
            return "Rolling std"
        return "Time features"

    imp_df["family"] = imp_df["feature"].apply(family)
    family_imp = imp_df.groupby("family")["importance"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top20 = imp_df.head(20)
    colors = [
        "#F44336" if feature == "lag_1"
        else "#2196F3" if feature.startswith("lag_")
        else "#FF9800" if feature.startswith("roll_")
        else "#4CAF50"
        for feature in top20["feature"]
    ]
    axes[0].barh(top20["feature"][::-1], top20["importance"][::-1], color=colors[::-1], alpha=0.85)
    axes[0].set_xlabel("Importance Score")
    axes[0].set_title("Top 20 Feature Importances (XGBoost)\nRed = lag_1", fontweight="bold")
    axes[0].legend(
        handles=[
            Patch(facecolor="#F44336", label="lag_1 (prev day)"),
            Patch(facecolor="#2196F3", label="Other lags"),
            Patch(facecolor="#FF9800", label="Rolling features"),
            Patch(facecolor="#4CAF50", label="Other"),
        ],
        fontsize=8,
    )

    wedge_colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]
    _, _, autotexts = axes[1].pie(
        family_imp.values,
        labels=family_imp.index,
        autopct="%1.1f%%",
        colors=wedge_colors[: len(family_imp)],
        startangle=140,
        pctdistance=0.75,
    )
    for text in autotexts:
        text.set_fontsize(9)
    axes[1].set_title("Feature Importance by Family\n(% of total model weight)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    lag1_pct = imp_df.loc[imp_df["feature"] == "lag_1", "importance"].iloc[0] * 100
    lag_total = imp_df[imp_df["feature"].str.startswith("lag_")]["importance"].sum() * 100
    print(f"  lag_1 alone accounts for {lag1_pct:.1f}% of model weight")
    print(f"  All lag features combined: {lag_total:.1f}%")
    print("  Saved feature_importance.png")


def residual_analysis() -> None:
    print("Residual Analysis")
    try:
        pred_df = pd.read_csv(TABLES_DIR / "LinearRegression_predictions.csv")
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    except FileNotFoundError:
        print("  Run train_timeseries.py first. Skipping residual analysis.")
        return

    pred_df["residual"] = pred_df["ActualPrice"] - pred_df["PredictedPrice"]
    pred_df["abs_residual"] = pred_df["residual"].abs()
    pred_df["pct_error"] = pred_df["abs_residual"] / pred_df["ActualPrice"] * 100

    district_res = (
        pred_df.groupby("District")
        .agg(
            MAE=("abs_residual", "mean"),
            RMSE=("residual", lambda x: np.sqrt((x**2).mean())),
            MeanPctError=("pct_error", "mean"),
            MaxError=("abs_residual", "max"),
        )
        .sort_values("MAE", ascending=False)
        .reset_index()
    )
    district_res.to_csv(TABLES_DIR / "district_residuals.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Residual Analysis - Linear Regression\n(where does the model fail?)", fontsize=13, fontweight="bold")

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(district_res)))
    axes[0, 0].barh(district_res["District"][::-1], district_res["MAE"][::-1], color=colors, alpha=0.85)
    axes[0, 0].set_xlabel("Mean Absolute Error (Million VND)")
    axes[0, 0].set_title("MAE by District", fontweight="bold")
    for i, mae in enumerate(district_res["MAE"][::-1]):
        axes[0, 0].text(mae + 0.02, i, f"{mae:.2f}", va="center", fontsize=8)

    axes[0, 1].barh(district_res["District"][::-1], district_res["MeanPctError"][::-1], color=colors, alpha=0.85)
    axes[0, 1].set_xlabel("Mean % Error")
    axes[0, 1].set_title("Mean Percentage Error by District", fontweight="bold")

    for district in pred_df["District"].unique():
        sub = pred_df[pred_df["District"] == district].sort_values("Date")
        axes[1, 0].plot(sub["Date"], sub["residual"], alpha=0.5, lw=0.8, label=district)
    axes[1, 0].axhline(0, color="black", lw=1, linestyle="--")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 0].set_title("Residuals Over Time by District", fontweight="bold")
    axes[1, 0].legend(ncol=3, fontsize=6)
    axes[1, 0].tick_params(axis="x", rotation=30, labelsize=7)

    axes[1, 1].hist(pred_df["residual"], bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(0, color="red", lw=1.5, linestyle="--", label="Zero residual")
    axes[1, 1].axvline(
        pred_df["residual"].mean(),
        color="orange",
        lw=1.5,
        linestyle="-",
        label=f"Mean: {pred_df['residual'].mean():.2f}",
    )
    axes[1, 1].set_xlabel("Residual (Million VND)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Residual Distribution (all districts)", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved residual_analysis.png")
    print(district_res.to_string(index=False))


def elbow_plot() -> None:
    print("K-Means Elbow Plot")
    df = load_modeling_data()
    summary = (
        df.groupby("District")
        .agg(avg_price=("Price", "mean"), std_price=("Price", "std"), min_price=("Price", "min"), max_price=("Price", "max"))
        .reset_index()
    )

    growth_rows = []
    for district, sub in df.groupby("District"):
        sub = sub.sort_values("Date")
        growth_rows.append({"District": district, "growth_rate": (sub.iloc[-1]["Price"] - sub.iloc[0]["Price"]) / sub.iloc[0]["Price"]})
    summary = summary.merge(pd.DataFrame(growth_rows), on="District")

    X = StandardScaler().fit_transform(summary[["avg_price", "std_price", "min_price", "max_price", "growth_rate"]])

    k_range = range(2, 9)
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_range), inertias, "o-", color="#2196F3", lw=2, ms=8)
    axes[0].axvline(3, color="red", lw=1.5, linestyle="--", label="k=3 (chosen)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Plot - K-Means Clustering", fontweight="bold")
    axes[0].legend()
    axes[0].set_xticks(list(k_range))

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    axes[1].plot(list(k_range), silhouettes, "o-", color="#FF5722", lw=2, ms=8)
    axes[1].axvline(3, color="red", lw=1.5, linestyle="--", label="k=3 (chosen)")
    axes[1].axvline(best_k, color="green", lw=1.5, linestyle="--", label=f"k={best_k} (best silhouette)")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score - K-Means Clustering", fontweight="bold")
    axes[1].legend()
    axes[1].set_xticks(list(k_range))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Best silhouette at k={best_k}; k=3 was chosen in the original analysis.")
    print("  Saved kmeans_elbow.png")


def price_gap_analysis() -> None:
    print("Price Gap / Convergence Analysis")
    df = load_modeling_data()

    daily = df.groupby("Date")["Price"].agg(price_max="max", price_min="min", price_mean="mean", price_std="std").reset_index()
    daily["price_gap"] = daily["price_max"] - daily["price_min"]
    daily["coeff_var"] = daily["price_std"] / daily["price_mean"]
    daily.to_csv(TABLES_DIR / "price_gap_trend.csv", index=False)

    daily["gap_smooth"] = daily["price_gap"].rolling(30, center=True).mean()
    daily["cv_smooth"] = daily["coeff_var"].rolling(30, center=True).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].fill_between(daily["Date"], daily["price_gap"], alpha=0.15, color="#2196F3")
    axes[0].plot(daily["Date"], daily["gap_smooth"], color="#2196F3", lw=2, label="30-day smoothed")
    axes[0].set_ylabel("Max - Min Price (Million VND)")
    axes[0].set_title("Price Gap Between Most & Least Expensive District Over Time", fontweight="bold")
    axes[0].legend()

    axes[1].fill_between(daily["Date"], daily["coeff_var"], alpha=0.15, color="#FF5722")
    axes[1].plot(daily["Date"], daily["cv_smooth"], color="#FF5722", lw=2, label="30-day smoothed")
    axes[1].set_ylabel("Coefficient of Variation (Std / Mean)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Normalised Price Dispersion - Are Districts Becoming More Equal?", fontweight="bold")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_gap_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    first_gap = daily["price_gap"].iloc[:30].mean()
    last_gap = daily["price_gap"].iloc[-30:].mean()
    direction = "widening (diverging)" if last_gap > first_gap else "narrowing (converging)"
    print(f"  Avg gap (first 30 days): {first_gap:.1f}M VND")
    print(f"  Avg gap (last 30 days): {last_gap:.1f}M VND - {direction}")
    print("  Saved price_gap_convergence.png")


def main() -> None:
    feature_importance_analysis()
    print()
    residual_analysis()
    print()
    elbow_plot()
    print()
    price_gap_analysis()
    print("\nAll advanced analyses complete.")


if __name__ == "__main__":
    main()

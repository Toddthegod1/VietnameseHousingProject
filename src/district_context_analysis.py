"""
Analyze district-level geographic and demographic context.

The context data adds two interpretable urban-economics features:
population density and distance to District 1, the central business district.
"""

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(a))


def load_context() -> pd.DataFrame:
    context = pd.read_csv(DATA_DIR / "district_context.csv")
    district1 = context.loc[context["District"] == "District 1"].iloc[0]

    context["population_density_per_km2"] = context["population"] / context["area_km2"]
    context["distance_to_district1_km"] = context.apply(
        lambda row: haversine_km(district1["lat"], district1["lon"], row["lat"], row["lon"]),
        axis=1,
    )
    return context


def build_context_summary() -> pd.DataFrame:
    context = load_context()
    district_summary = pd.read_csv(TABLES_DIR / "district_summary.csv")

    summary = district_summary.merge(context, on="District", how="left")
    summary = summary.sort_values("District").reset_index(drop=True)
    summary.to_csv(TABLES_DIR / "district_context_summary.csv", index=False)
    print("Saved district_context_summary.csv")
    return summary


def annotate_points(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            row["District"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )


def add_fit_line(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    if len(np.unique(x)) < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", lw=1, linestyle="--", alpha=0.7)


def plot_price_context(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (
            axes[0],
            "distance_to_district1_km",
            "avg_price",
            "Distance to District 1 (km)",
            "Average Price (Million VND)",
            "Average Price vs Distance to CBD",
            "#4C78A8",
        ),
        (
            axes[1],
            "population_density_per_km2",
            "avg_price",
            "Population Density (people/km2)",
            "Average Price (Million VND)",
            "Average Price vs Population Density",
            "#F58518",
        ),
    ]

    for ax, x_col, y_col, xlabel, ylabel, title, color in panels:
        ax.scatter(summary[x_col], summary[y_col], s=90, color=color, alpha=0.85)
        annotate_points(ax, summary, x_col, y_col)
        add_fit_line(ax, summary, x_col, y_col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_context_price_relationships.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved district_context_price_relationships.png")


def plot_growth_context(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (
            axes[0],
            "distance_to_district1_km",
            "growth_rate",
            "Distance to District 1 (km)",
            "Cumulative Growth Rate",
            "Growth vs Distance to CBD",
            "#54A24B",
        ),
        (
            axes[1],
            "population_density_per_km2",
            "growth_rate",
            "Population Density (people/km2)",
            "Cumulative Growth Rate",
            "Growth vs Population Density",
            "#E45756",
        ),
    ]

    for ax, x_col, y_col, xlabel, ylabel, title, color in panels:
        ax.scatter(summary[x_col], summary[y_col], s=90, color=color, alpha=0.85)
        annotate_points(ax, summary, x_col, y_col)
        add_fit_line(ax, summary, x_col, y_col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_context_growth_relationships.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved district_context_growth_relationships.png")


def plot_correlation_heatmap(summary: pd.DataFrame) -> None:
    corr_cols = [
        "avg_price",
        "std_price",
        "growth_rate",
        "area_km2",
        "population",
        "population_density_per_km2",
        "distance_to_district1_km",
    ]
    corr = summary[corr_cols].corr()
    corr.to_csv(TABLES_DIR / "district_context_correlations.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("District Context Correlations", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_context_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved district_context_correlations.png")


def main() -> None:
    summary = build_context_summary()
    plot_price_context(summary)
    plot_growth_context(summary)
    plot_correlation_heatmap(summary)

    print("\nDistrict Context Summary")
    display_cols = [
        "District",
        "area_km2",
        "population",
        "population_density_per_km2",
        "distance_to_district1_km",
        "avg_price",
        "growth_rate",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

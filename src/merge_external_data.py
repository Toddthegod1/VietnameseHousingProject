"""
Merge external CPI, population, and HCM spatial cost data into the housing
dataset. Produces real prices, affordability metrics, and external-data plots.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_housing() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "HousePricingHCM.csv")
    df.columns = [c.strip() for c in df.columns]

    long = df.melt(id_vars="Date", var_name="District", value_name="Price")
    long["Date"] = pd.to_datetime(long["Date"])
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    long = long.dropna(subset=["Price"]).sort_values(["District", "Date"]).reset_index(drop=True)
    long["year"] = long["Date"].dt.year
    long["month"] = long["Date"].dt.month
    return long


def load_cpi() -> pd.DataFrame:
    with open(DATA_DIR / "consumerPriceIndex.csv", "rb") as f:
        text = f.read().decode("utf-8", errors="replace").replace("\x00", "")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = list(csv.reader(lines))
    header = rows[1]
    values = rows[2]

    year_cols = [(i, int(value)) for i, value in enumerate(header) if value.isdigit()]
    cpi = pd.DataFrame(
        {
            "year": [year for _, year in year_cols],
            "cpi_monthly_avg": [float(values[i]) for i, _ in year_cols],
        }
    )

    base_val = cpi.loc[cpi["year"] == 2017, "cpi_monthly_avg"].iloc[0]
    cpi["cpi_index"] = cpi["cpi_monthly_avg"] / base_val * 100
    cpi["annual_multiplier"] = (cpi["cpi_monthly_avg"] / 100) ** 12

    cpi = cpi.sort_values("year").reset_index(drop=True)
    base_pos = cpi.index[cpi["year"] == 2017][0]
    cpi["cum_price_level"] = 1.0

    for i in range(base_pos + 1, len(cpi)):
        cpi.loc[i, "cum_price_level"] = (
            cpi.loc[i - 1, "cum_price_level"] * cpi.loc[i, "annual_multiplier"]
        )

    for i in range(base_pos - 1, -1, -1):
        cpi.loc[i, "cum_price_level"] = (
            cpi.loc[i + 1, "cum_price_level"] / cpi.loc[i + 1, "annual_multiplier"]
        )

    return cpi[["year", "cpi_monthly_avg", "cpi_index", "cum_price_level"]]


def load_population() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "populationData.csv", skiprows=4, header=0)
    vnm = raw[raw["Country Code"] == "VNM"].copy()

    year_cols = [c for c in vnm.columns if c.strip().isdigit()]
    pop_long = vnm[year_cols].T.reset_index()
    pop_long.columns = ["year", "population"]
    pop_long["year"] = pop_long["year"].astype(int)
    pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")
    pop_long = pop_long.dropna(subset=["population"]).sort_values("year")
    pop_long["pop_growth_rate"] = pop_long["population"].pct_change()

    return pop_long[["year", "population", "pop_growth_rate"]]


def load_spatial_cost() -> pd.DataFrame:
    raw = pd.read_excel(DATA_DIR / "hochiSpatialcost.xlsx", header=None)
    years = [int(v) for v in raw.iloc[2, 1:].dropna()]
    values = [float(v) for v in raw.iloc[3, 1:].dropna()]
    return pd.DataFrame({"year": years, "hcm_spatial_cost_index": values})


def build_enriched_dataset() -> pd.DataFrame:
    housing = load_housing()
    cpi = load_cpi()
    pop = load_population()
    spatial = load_spatial_cost()

    df = housing.merge(cpi, on="year", how="left")
    df = df.merge(pop, on="year", how="left")
    df = df.merge(spatial, on="year", how="left")

    df["real_price"] = df["Price"] / df["cum_price_level"]
    df["population_millions"] = df["population"] / 1e6
    df["affordability_index"] = df["Price"] / df["cpi_index"] * 100

    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    df.to_csv(TABLES_DIR / "enriched_dataset.csv", index=False)
    print(f"Enriched dataset saved: {len(df):,} rows, {df.shape[1]} columns.")
    return df


def plot_real_vs_nominal(df: pd.DataFrame) -> None:
    districts = sorted(df["District"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    fig.suptitle(
        "Nominal vs Real (Inflation-Adjusted) Housing Prices\n"
        "Ho Chi Minh City Districts - 2017 VND Base",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for ax, district, color in zip(axes.flat, districts, colors):
        sub = df[df["District"] == district].sort_values("Date")
        ax.plot(sub["Date"], sub["Price"], label="Nominal", color=color, lw=1.5)
        ax.plot(sub["Date"], sub["real_price"], label="Real", color=color, lw=1.5, linestyle="--", alpha=0.7)
        ax.set_title(district, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if ax == axes.flat[0]:
            ax.legend(fontsize=7)

    fig.supxlabel("Date", fontsize=11)
    fig.supylabel("Price (Millions VND)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_vs_nominal_prices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved real_vs_nominal_prices.png")


def plot_real_growth_comparison(df: pd.DataFrame) -> None:
    districts = sorted(df["District"].unique())
    nominal_growth = []
    real_growth = []

    for district in districts:
        sub = df[df["District"] == district].sort_values("Date")
        nominal_growth.append((sub["Price"].iloc[-1] / sub["Price"].iloc[0] - 1) * 100)
        real_growth.append((sub["real_price"].iloc[-1] / sub["real_price"].iloc[0] - 1) * 100)

    x = np.arange(len(districts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, nominal_growth, width, label="Nominal Growth (%)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, real_growth, width, label="Real Growth (%)", color="#FF5722", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(districts, rotation=30, ha="right")
    ax.set_ylabel("Cumulative Growth (%)")
    ax.set_title(
        "Nominal vs Real Cumulative Housing Price Growth by District\n"
        "(Inflation-adjusted, 2017 base)",
        fontweight="bold",
    )
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_vs_nominal_growth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved real_vs_nominal_growth.png")


def plot_affordability_trend(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for district in sorted(df["District"].unique()):
        sub = df[df["District"] == district].sort_values("Date")
        ax.plot(sub["Date"], sub["affordability_index"], label=district, lw=1.5)

    ax.set_title(
        "Affordability Index by District Over Time\n"
        "(Price / CPI Index x 100 - rising = less affordable)",
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Affordability Index")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "affordability_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved affordability_trend.png")


def plot_population_vs_price(df: pd.DataFrame) -> None:
    annual = (
        df.groupby(["District", "year"])
        .agg(avg_real_price=("real_price", "mean"), pop_growth_rate=("pop_growth_rate", "first"))
        .reset_index()
    )
    price_growth = (
        annual.groupby("District")
        .apply(lambda g: g.sort_values("year")["avg_real_price"].pct_change().mean())
        .rename("avg_annual_real_price_growth")
        .reset_index()
    )
    pop_g = annual.groupby("District")["pop_growth_rate"].mean().reset_index()
    merged = price_growth.merge(pop_g, on="District")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    for i, row in merged.iterrows():
        ax.scatter(
            row["pop_growth_rate"] * 100,
            row["avg_annual_real_price_growth"] * 100,
            color=colors[i % 10],
            s=120,
            zorder=3,
        )
        ax.annotate(
            row["District"],
            (row["pop_growth_rate"] * 100, row["avg_annual_real_price_growth"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.axvline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlabel("Vietnam Annual Population Growth Rate (%)")
    ax.set_ylabel("District Avg Annual Real Price Growth (%)")
    ax.set_title(
        "Real Price Growth vs. National Population Growth\n"
        "(each point = one district average across all years)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "population_vs_real_price_growth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved population_vs_real_price_growth.png")


def main() -> None:
    df = build_enriched_dataset()
    plot_real_vs_nominal(df)
    plot_real_growth_comparison(df)
    plot_affordability_trend(df)
    plot_population_vs_price(df)
    print("\nAll external data plots saved.")


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_predictions(model_name):
    df = pd.read_csv(TABLES_DIR / f"{model_name}_predictions.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    sample_districts = df["District"].drop_duplicates().tolist()[:4]

    for district in sample_districts:
        sub = df[df["District"] == district].sort_values("Date")

        plt.figure(figsize=(10, 5))
        plt.plot(sub["Date"], sub["ActualPrice"], label="Actual")
        plt.plot(sub["Date"], sub["PredictedPrice"], label="Predicted")
        plt.title(f"{model_name}: Actual vs Predicted for {district}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{model_name}_{district}_timeseries.png", dpi=300)
        plt.close()


def plot_average_price_by_district():
    df = pd.read_csv(TABLES_DIR / "modeling_dataset.csv")

    avg_df = (
        df.groupby("District")["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    avg_df.plot(kind="bar")
    plt.title("Average Housing Price by District")
    plt.xlabel("District")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "avg_price_by_district.png", dpi=300)
    plt.close()


def plot_price_trends():
    df = pd.read_csv(TABLES_DIR / "modeling_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    sample_districts = df["District"].drop_duplicates().tolist()[:6]

    plt.figure(figsize=(12, 6))
    for district in sample_districts:
        sub = df[df["District"] == district].sort_values("Date")
        plt.plot(sub["Date"], sub["Price"], label=district)

    plt.title("Housing Price Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_price_trends.png", dpi=300)
    plt.close()


def main():
    plot_average_price_by_district()
    plot_price_trends()

    for model_name in ["LinearRegression", "RandomForest", "XGBoost"]:
        plot_model_predictions(model_name)

    print("Saved evaluation plots.")


if __name__ == "__main__":
    main()
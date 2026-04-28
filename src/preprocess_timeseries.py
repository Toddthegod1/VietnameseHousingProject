from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "HousePricingHCM.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def load_and_reshape_data():
    df = pd.read_csv(DATA_FILE)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Convert wide -> long
    long_df = df.melt(
        id_vars="Date",
        var_name="District",
        value_name="Price"
    )

    long_df["Date"] = pd.to_datetime(long_df["Date"])
    long_df["Price"] = pd.to_numeric(long_df["Price"], errors="coerce")

    # Drop bad rows
    long_df = long_df.dropna(subset=["Price"])
    long_df = long_df.sort_values(["District", "Date"]).reset_index(drop=True)

    return long_df


def add_time_features(df):
    df = df.copy()

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["quarter"] = df["Date"].dt.quarter

    return df


def add_lag_features(df, lags=(1, 2, 3, 7, 14, 30)):
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("District")["Price"].shift(lag)

    return df


def add_rolling_features(df, windows=(3, 7, 14, 30)):
    df = df.copy()

    for window in windows:
        df[f"roll_mean_{window}"] = (
            df.groupby("District")["Price"]
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f"roll_std_{window}"] = (
            df.groupby("District")["Price"]
            .transform(lambda x: x.shift(1).rolling(window).std())
        )

    return df


def build_modeling_dataset():
    df = load_and_reshape_data()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Remove rows where lag/rolling features are missing
    df = df.dropna().reset_index(drop=True)

    df.to_csv(TABLES_DIR / "modeling_dataset.csv", index=False)
    print("Saved modeling dataset.")
    print(df.head())

    return df


if __name__ == "__main__":
    build_modeling_dataset()

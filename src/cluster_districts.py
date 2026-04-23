from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_district_summary():
    df = pd.read_csv(TABLES_DIR / "modeling_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    summary = df.groupby("District").agg(
        avg_price=("Price", "mean"),
        std_price=("Price", "std"),
        min_price=("Price", "min"),
        max_price=("Price", "max")
    ).reset_index()

    # Growth estimate
    growth_rates = []
    for district, sub in df.groupby("District"):
        sub = sub.sort_values("Date")
        first_price = sub.iloc[0]["Price"]
        last_price = sub.iloc[-1]["Price"]
        growth = (last_price - first_price) / first_price if first_price != 0 else 0
        growth_rates.append((district, growth))

    growth_df = pd.DataFrame(growth_rates, columns=["District", "growth_rate"])
    summary = summary.merge(growth_df, on="District", how="left")

    summary.to_csv(TABLES_DIR / "district_summary.csv", index=False)
    return summary


def cluster_districts():
    summary = build_district_summary()

    feature_cols = ["avg_price", "std_price", "min_price", "max_price", "growth_rate"]
    X = summary[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    summary["cluster"] = kmeans.fit_predict(X_scaled)

    summary.to_csv(TABLES_DIR / "district_clusters.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=summary["cluster"])
    for i, district in enumerate(summary["District"]):
        plt.text(X_pca[i, 0], X_pca[i, 1], district, fontsize=8)

    plt.title("District Clusters Based on Housing Price Patterns")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "district_clusters.png", dpi=300)
    plt.close()

    print(summary[["District", "cluster", "avg_price", "growth_rate"]])


if __name__ == "__main__":
    cluster_districts()
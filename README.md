# Vietnamese Housing Project

Our project analyzes district-level housing prices in Ho Chi Minh City over time. We transform date-based district price data into a time-series machine learning dataset, engineer lag and rolling statistical features, and train regression models to predict future housing prices. We evaluate model performance using MAE, RMSE, and R², and we further analyze district-level differences through trend visualization and clustering.

## Project Structure

- `data/`: Contains the raw data file `HousePricingHCM.csv`
- `src/`: Source code for preprocessing, training, evaluation, and clustering
- `outputs/`: Generated outputs including figures, tables, and models
- `run_pipeline.py`: Script to run the entire pipeline
- `requirements.txt`: Python dependencies

## Team Roles

- **Emerson**: Clean the raw CSV, verify dates and district columns, handle missing values
- **Will**: Train linear regression, random forest, XGBoost models, compare performance, tune XGBoost
- **Todd**: Build all plots, compare district trends, make clustering visuals, write interpretation section

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python run_pipeline.py`

## Key Insights

### Housing Price Distribution
- **Highest-priced districts**: District 1 (avg. $193), District 3 ($160), District 5 ($157)
- **Lowest-priced districts**: District 9 (avg. $50), District 2 ($67)
- **Price range**: $12.22 (District 9 min) to $286.93 (District 1 max)

### Growth Patterns
- **Fastest growing**: District 7 with 3.5x growth - emerging hotspot
- **Slowest growing**: District 6 with 0.32x growth - stable market
- **Strong growth districts**: Districts 9 (1.53x), 5 (1.39x), 1 (1.29x)

### District Clustering (K-means, k=3)
- **Cluster 1 (Premium growing)**: Districts 1, 3, 5 - high-value established markets
- **Cluster 0 (Affordable moderate)**: Districts 2, 4, 6, 8, 9 - varied growth patterns
- **Cluster 2 (Outlier)**: District 7 - low price but explosive growth

### Model Performance
- **Excellent accuracy** (R² > 99% for all models)
- **Best performer**: Linear Regression (MAE: 1.53, RMSE: 1.90)
- **All districts highly predictable** using time-series features

### Strategic Conclusions
District-level temporal patterns are powerful predictors of future housing prices. District 7 represents the greatest growth opportunity, while Districts 1, 3, and 5 are established premium markets.
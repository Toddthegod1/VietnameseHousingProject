# Vietnamese Housing Project

This project analyzes district-level housing prices in Ho Chi Minh City over time. It reshapes raw district price data into a time-series modeling dataset, trains several forecasting models, compares short-term and longer-horizon prediction accuracy, incorporates external economic data, and studies how district housing markets differ.

All housing prices in this project are measured in **millions of Vietnamese Dong (VND)**.

## Project Structure

- `data/`: Raw input data
  - `HousePricingHCM.csv`: district-level housing price time series
  - `consumerPriceIndex.csv`: CPI data used for inflation adjustment
  - `populationData.csv`: Vietnam population data
  - `hochiSpatialcost.xlsx`: HCM spatial cost index data
  - `district_context.csv`: district area, population, approximate coordinates, population density inputs
- `src/`: Python scripts for preprocessing, modeling, forecasting, and analysis
- `outputs/figures/`: Generated PNG plots
- `outputs/tables/`: Generated CSV tables
- `outputs/models/`: Generated `.joblib` model files, ignored by Git because they are large and reproducible
- `run_pipeline.py`: Runs the full analysis pipeline
- `requirements.txt`: Python dependencies

## Main Pipeline

Run the complete project with:

```bash
python run_pipeline.py
```

The pipeline runs these scripts in order:

1. `src/preprocess_timeseries.py`
   - Converts the housing data from wide to long format
   - Adds time, lag, and rolling-window features
   - Saves `outputs/tables/modeling_dataset.csv`

2. `src/train_timeseries.py`
   - Trains Linear Regression, Random Forest, and XGBoost models
   - Uses a time-based train/test split, not a random split
   - Saves model metrics and predictions

3. `src/evaluate_timeseries.py`
   - Generates model evaluation plots

4. `src/cluster_districts.py`
   - Creates district-level summary statistics
   - Runs K-means clustering and PCA visualization

5. `src/district_context_analysis.py`
   - Adds district population density and distance to District 1
   - Compares these context variables against average price and growth

6. `src/merge_external_data.py`
   - Merges CPI, population, and spatial cost data with the housing data
   - Creates real inflation-adjusted prices and affordability metrics

7. `src/multi_horizon_forecast.py`
   - Compares 1-day, 7-day, and 30-day price-level forecasts

8. `src/predict_growth_30d.py`
   - Predicts 30-day percentage price growth by district
   - This is a harder and more decision-useful target than next-day price level

9. `src/advanced_analysis.py`
   - Generates feature importance, residual analysis, K-means elbow/silhouette plots, and price-gap convergence analysis

## Current Model Results

The standard price-level models are very accurate, but this should be interpreted carefully because housing prices are highly autocorrelated. In other words, yesterday's price is very useful for predicting today's or tomorrow's price.

From `outputs/tables/model_results.csv`:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 1.53 | 1.90 | 0.9989 |
| XGBoost | 1.58 | 2.09 | 0.9986 |
| Random Forest | 1.79 | 2.59 | 0.9979 |

The high R2 values are not just a win; they are also a warning. Feature importance shows that `lag_1` alone accounts for about **51.9%** of XGBoost feature importance, meaning the model is heavily relying on the previous price.

## Multi-Horizon Forecasting

To test whether the model is doing more than repeating recent prices, the project also forecasts prices farther ahead.

From `outputs/tables/multi_horizon_overall.csv`:

| Horizon | MAE | RMSE | R2 |
|---:|---:|---:|---:|
| 1 day | 1.65 | 2.16 | 0.9985 |
| 7 days | 2.05 | 2.84 | 0.9975 |
| 30 days | 2.79 | 4.01 | 0.9950 |

Accuracy gets worse as the forecast horizon increases. This supports the main modeling interpretation: short-term price levels are easy to predict because prices move smoothly, but longer-horizon forecasting is harder.

## 30-Day Growth Forecast

The project now includes a more difficult target: predicting **30-day percentage price growth** by district.

From `outputs/tables/growth_30d_overall_metrics.csv`:

| Model | Horizon | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| XGBoost | 30 days | 2.17 percentage points | 2.84 percentage points | -0.040 |
| Naive zero-growth baseline | 30 days | 2.18 percentage points | 2.78 percentage points | ~0.000 |
| Naive past-growth baseline | 30 days | 3.31 percentage points | 4.19 percentage points | -1.267 |

This result is important: predicting growth is much harder than predicting price level. XGBoost only slightly improves MAE over a simple zero-growth baseline, so the project should not claim that 30-day appreciation is strongly predictable.

Latest available 30-day growth ranking from `outputs/tables/growth_30d_latest_district_forecast.csv`:

| Rank | District | Predicted 30-Day Growth |
|---:|---|---:|
| 1 | District 2 | 3.64% |
| 2 | District 8 | 3.60% |
| 3 | District 9 | 2.67% |
| 4 | District 3 | 1.60% |
| 5 | District 5 | 1.37% |
| 6 | District 1 | 0.67% |
| 7 | District 7 | 0.22% |
| 8 | District 6 | 0.09% |
| 9 | District 4 | -0.52% |

## External Data Analysis

The external-data script adds economic context:

- CPI is used to calculate real, inflation-adjusted housing prices.
- Population data is merged by year to add demographic context.
- Spatial cost data is merged by year for HCM cost comparisons.
- The project creates affordability and real-price plots.

Key output files:

- `outputs/tables/enriched_dataset.csv`
- `outputs/figures/real_vs_nominal_prices.png`
- `outputs/figures/real_vs_nominal_growth.png`
- `outputs/figures/affordability_trend.png`
- `outputs/figures/population_vs_real_price_growth.png`

## District-Level Findings

From `outputs/tables/district_summary.csv`:

- Highest average prices: District 1, District 3, District 5
- Lowest average prices: District 9, District 2, District 7
- Fastest cumulative growth: District 7
- Slowest cumulative growth: District 6

The clustering analysis currently uses `k=3`:

- Cluster 1: Districts 1, 3, 5
- Cluster 0: Districts 2, 4, 6, 8, 9
- Cluster 2: District 7

However, the advanced analysis shows that the silhouette score is strongest at `k=2`, suggesting the data may naturally split more cleanly into a premium group and a more affordable group.

## District Context Features

The project now includes `data/district_context.csv`, which adds:

- district land area
- district population
- population density
- approximate latitude/longitude
- straight-line distance to District 1

These features are analyzed in `src/district_context_analysis.py`, which produces:

- `outputs/tables/district_context_summary.csv`
- `outputs/tables/district_context_correlations.csv`
- `outputs/figures/district_context_price_relationships.png`
- `outputs/figures/district_context_growth_relationships.png`
- `outputs/figures/district_context_correlations.png`

Important caveat: District 2 and District 9 were merged into Thu Duc City in 2021, but the original housing dataset still uses old District 2 and District 9 labels. The context file keeps those former-district labels for compatibility. Source notes are documented in `data/district_context_sources.md`.

## Main Interpretation

The strongest conclusion is not simply that the models achieve very high R2. A more accurate interpretation is:

> Ho Chi Minh City housing prices are highly autocorrelated, so short-term price levels are easy to forecast from recent prices. However, predicting future appreciation is much harder. Inflation adjustment reduces the apparent size of nominal growth, and district-level price gaps have widened over time, suggesting increasing spatial inequality across the housing market.

## Notes for GitHub

Generated model files in `outputs/models/` are intentionally ignored because they are large and can be recreated by running the pipeline. Generated figures and tables are included so the results can be inspected without rerunning everything.

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "outputs" / "tables"
MODELS_DIR = BASE_DIR / "outputs" / "models"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_modeling_data():
    df = pd.read_csv(TABLES_DIR / "modeling_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def time_train_test_split(df):
    # Use time-based split, not random split
    split_date = df["Date"].quantile(0.8)

    train_df = df[df["Date"] <= split_date].copy()
    test_df = df[df["Date"] > split_date].copy()

    return train_df, test_df


def main():
    df = load_modeling_data()
    train_df, test_df = time_train_test_split(df)

    target = "Price"
    drop_cols = ["Date", "Price"]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]

    categorical_features = ["District"]
    numeric_features = [col for col in X_train.columns if col != "District"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        ),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        results.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse(y_test, preds),
            "R2": r2_score(y_test, preds)
        })

        pred_df = test_df[["Date", "District"]].copy()
        pred_df["ActualPrice"] = y_test.values
        pred_df["PredictedPrice"] = preds
        pred_df.to_csv(TABLES_DIR / f"{model_name}_predictions.csv", index=False)

        joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(TABLES_DIR / "model_results.csv", index=False)

    print(results_df)


if __name__ == "__main__":
    main()

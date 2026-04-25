from __future__ import annotations

from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.project_config import ROOT_DIR, SEED

TARGET_COLUMN = "price_rub"
BASE_FEATURE_COLUMNS = [
    "brand",
    "screen_diagonal_inch",
    "screen_resolution",
    "matrix_type",
    "cpu",
    "gpu",
    "ram_gb",
    "ram_type",
    "storage_gb",
    "os",
    "weight_kg",
    "resolution_width",
    "resolution_height",
    "screen_pixels_mln",
    "cpu_vendor",
    "gpu_vendor",
    "is_discrete_gpu",
    "is_without_os",
    "is_gaming_series",
]

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "baseline_ridge.joblib"
DEFAULT_LINREG_MODEL_PATH = ROOT_DIR / "models" / "linreg_raw.joblib"

RAW_NUMERIC_COLUMNS = ["screen_diagonal_inch", "ram_gb", "storage_gb", "weight_kg"]
RAW_CATEGORICAL_COLUMNS = ["brand", "matrix_type", "screen_resolution", "cpu", "gpu", "ram_type", "os"]
RAW_FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS


def split_dataset(dataframe: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    usable = dataframe.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    train_val, test = train_test_split(usable, test_size=0.15, random_state=seed)
    validation_ratio = 0.15 / 0.85
    train, validation = train_test_split(train_val, test_size=validation_ratio, random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def infer_feature_groups(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    available_columns = [column for column in BASE_FEATURE_COLUMNS if column in dataframe.columns]
    numeric_columns = [column for column in available_columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    categorical_columns = [column for column in available_columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_columns, categorical_columns = infer_feature_groups(dataframe)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_dummy_regressor() -> DummyRegressor:
    return DummyRegressor(strategy="median")


def build_linreg_raw_pipeline() -> Pipeline:
    """LinearRegression on raw features only — no feature engineering."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, RAW_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, RAW_CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def build_baseline_pipeline(dataframe: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(dataframe)),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def evaluate_regression(y_true: pd.Series, predictions: pd.Series | list[float]) -> dict[str, float]:
    mae = mean_absolute_error(y_true, predictions)
    rmse = sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def fit_and_score(model, train_df: pd.DataFrame, evaluation_df: pd.DataFrame) -> dict[str, object]:
    feature_columns = [column for column in BASE_FEATURE_COLUMNS if column in train_df.columns]
    model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    predictions = model.predict(evaluation_df[feature_columns])
    metrics = evaluate_regression(evaluation_df[TARGET_COLUMN], predictions)
    return {"model": model, "predictions": predictions, "metrics": metrics}


def save_model(model: Pipeline, output_path: Path | None = None) -> Path:
    path = output_path or DEFAULT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

"""CP2: model selection experiments.

Trains a set of models on the same train/val/test split, runs hyperparameter
search where applicable, records metrics, picks the best model by validation
MAE, refits on train+val, evaluates on test, and saves:
  - models/experiments.csv         flat table for the report
  - models/experiments.json        full metrics + tuned hyperparameters
  - models/final_model.joblib      best pipeline refit on train+val
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.modeling import (
    BASE_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_regression,
    split_dataset,
)
from src.preprocessing import clean_laptops_dataframe, load_raw_dataset
from src.project_config import ROOT_DIR, SEED

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENTS_CSV = ROOT_DIR / "models" / "experiments.csv"
EXPERIMENTS_JSON = ROOT_DIR / "models" / "experiments.json"
FINAL_MODEL_PATH = ROOT_DIR / "models" / "final_model.joblib"


def _split_feature_groups(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    available = [c for c in columns if c in df.columns]
    numeric = [c for c in available if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in available if c not in numeric]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, columns: list[str], dense: bool = False) -> ColumnTransformer:
    numeric, categorical = _split_feature_groups(df, columns)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
                    ]
                ),
                categorical,
            ),
        ]
    )


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    feature_columns: list[str]
    pipeline_factory: Callable[[], Pipeline]
    param_grid: dict[str, Any] | None = None
    search: str = "grid"
    n_iter: int = 15


@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    feature_columns: list[str]
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0


def _fit_one(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> ExperimentResult:
    X_train, y_train = train_df[config.feature_columns], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[config.feature_columns], val_df[TARGET_COLUMN]
    pipeline = config.pipeline_factory()

    started = time.perf_counter()
    if config.param_grid:
        if config.search == "random":
            searcher: Any = RandomizedSearchCV(
                pipeline,
                param_distributions=config.param_grid,
                n_iter=config.n_iter,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                random_state=SEED,
            )
        else:
            searcher = GridSearchCV(
                pipeline,
                param_grid=config.param_grid,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
        searcher.fit(X_train, y_train)
        estimator = searcher.best_estimator_
        params = {k: _to_jsonable(v) for k, v in searcher.best_params_.items()}
    else:
        pipeline.fit(X_train, y_train)
        estimator = pipeline
        params = {}
    fit_seconds = time.perf_counter() - started

    val_metrics = evaluate_regression(y_val, estimator.predict(X_val))
    return ExperimentResult(
        name=config.name,
        hypothesis=config.hypothesis,
        feature_columns=config.feature_columns,
        estimator=estimator,
        params=params,
        val_metrics=val_metrics,
        fit_seconds=fit_seconds,
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def define_experiments(train_df: pd.DataFrame) -> list[ExperimentConfig]:
    fe_features = [c for c in BASE_FEATURE_COLUMNS if c in train_df.columns]
    raw_features = [c for c in RAW_FEATURE_COLUMNS if c in train_df.columns]

    def linreg_raw_pipe() -> Pipeline:
        return Pipeline([("pre", build_preprocessor(train_df, raw_features)), ("model", LinearRegression())])

    def linreg_fe_pipe() -> Pipeline:
        return Pipeline([("pre", build_preprocessor(train_df, fe_features)), ("model", LinearRegression())])

    def ridge_pipe() -> Pipeline:
        return Pipeline([("pre", build_preprocessor(train_df, fe_features)), ("model", Ridge())])

    def lasso_pipe() -> Pipeline:
        return Pipeline([("pre", build_preprocessor(train_df, fe_features)), ("model", Lasso(max_iter=20000))])

    def knn_pipe() -> Pipeline:
        return Pipeline(
            [("pre", build_preprocessor(train_df, fe_features, dense=True)), ("model", KNeighborsRegressor())]
        )

    def tree_pipe() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(train_df, fe_features, dense=True)),
                ("model", DecisionTreeRegressor(random_state=SEED)),
            ]
        )

    def rf_pipe() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(train_df, fe_features, dense=True)),
                ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1)),
            ]
        )

    def gb_pipe() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(train_df, fe_features, dense=True)),
                (
                    "model",
                    GradientBoostingRegressor(random_state=SEED, n_estimators=400, max_depth=4, learning_rate=0.05),
                ),
            ]
        )

    def lgbm_pipe() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(train_df, fe_features, dense=True)),
                ("model", LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)),
            ]
        )

    def ridge_pca_pipe() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(train_df, fe_features, dense=True)),
                ("pca", PCA(n_components=0.95, random_state=SEED)),
                ("model", Ridge(alpha=10.0)),
            ]
        )

    return [
        ExperimentConfig(
            name="LinearRegression_raw",
            hypothesis="Линейная модель на сырых признаках без FE",
            feature_columns=raw_features,
            pipeline_factory=linreg_raw_pipe,
        ),
        ExperimentConfig(
            name="LinearRegression_FE",
            hypothesis="Те же FE-фичи помогают линейной модели",
            feature_columns=fe_features,
            pipeline_factory=linreg_fe_pipe,
        ),
        ExperimentConfig(
            name="Ridge_FE_grid",
            hypothesis="L2 уменьшает variance на разреженной OneHot-матрице",
            feature_columns=fe_features,
            pipeline_factory=ridge_pipe,
            param_grid={"model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
            search="grid",
        ),
        ExperimentConfig(
            name="Lasso_FE_grid",
            hypothesis="L1 делает отбор признаков",
            feature_columns=fe_features,
            pipeline_factory=lasso_pipe,
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]},
            search="grid",
        ),
        ExperimentConfig(
            name="KNN_FE_grid",
            hypothesis="Похожие по характеристикам ноуты стоят похоже",
            feature_columns=fe_features,
            pipeline_factory=knn_pipe,
            param_grid={
                "model__n_neighbors": [3, 5, 7, 10, 15],
                "model__weights": ["uniform", "distance"],
            },
            search="grid",
        ),
        ExperimentConfig(
            name="DecisionTree_FE_grid",
            hypothesis="Одно дерево ловит нелинейности",
            feature_columns=fe_features,
            pipeline_factory=tree_pipe,
            param_grid={
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 10],
            },
            search="grid",
        ),
        ExperimentConfig(
            name="RandomForest_random",
            hypothesis="Bagging-ансамбль усредняет шум одиночного дерева",
            feature_columns=fe_features,
            pipeline_factory=rf_pipe,
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12, 20],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", 0.5],
            },
            search="random",
            n_iter=10,
        ),
        ExperimentConfig(
            name="GradientBoosting",
            hypothesis="Sequential boosting лечит ошибки baseline",
            feature_columns=fe_features,
            pipeline_factory=gb_pipe,
        ),
        ExperimentConfig(
            name="LightGBM_random",
            hypothesis="Современный GBM — обычно лидер на табличных данных",
            feature_columns=fe_features,
            pipeline_factory=lgbm_pipe,
            param_grid={
                "model__n_estimators": [200, 400, 800],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [-1, 6, 10],
                "model__num_leaves": [31, 63, 127],
                "model__min_child_samples": [10, 20, 40],
            },
            search="random",
            n_iter=15,
        ),
        ExperimentConfig(
            name="Ridge_PCA95",
            hypothesis="PCA убирает корреляцию OneHot-фич, сохраняя 95% дисперсии",
            feature_columns=fe_features,
            pipeline_factory=ridge_pca_pipe,
        ),
    ]


def run_all(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for config in define_experiments(train_df):
        print(f"  [{config.name}] {config.hypothesis}")
        result = _fit_one(config, train_df, val_df)
        print(
            f"    MAE_val={result.val_metrics['mae']:.0f}  "
            f"R2_val={result.val_metrics['r2']:.3f}  ({result.fit_seconds:.1f}s)"
        )
        results.append(result)
    return results


def results_to_table(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "hypothesis": r.hypothesis,
                "n_features": len(r.feature_columns),
                "best_params": json.dumps(r.params, ensure_ascii=False),
                "mae_val": round(r.val_metrics["mae"], 1),
                "rmse_val": round(r.val_metrics["rmse"], 1),
                "r2_val": round(r.val_metrics["r2"], 4),
                "fit_seconds": round(r.fit_seconds, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_val").reset_index(drop=True)


def select_and_finalize(
    results: list[ExperimentResult],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    best = min(results, key=lambda r: r.val_metrics["mae"])
    print(f"\nBest by val MAE: {best.name} (MAE={best.val_metrics['mae']:.0f})")

    train_val = pd.concat([train_df, val_df], ignore_index=True)
    final_pipeline = best.estimator
    final_pipeline.fit(train_val[best.feature_columns], train_val[TARGET_COLUMN])
    test_predictions = final_pipeline.predict(test_df[best.feature_columns])
    test_metrics = evaluate_regression(test_df[TARGET_COLUMN], test_predictions)
    print(f"Test:  MAE={test_metrics['mae']:.0f}  R2={test_metrics['r2']:.3f}")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, FINAL_MODEL_PATH)
    print(f"Saved final model -> {FINAL_MODEL_PATH}")

    return {
        "name": best.name,
        "hypothesis": best.hypothesis,
        "feature_columns": best.feature_columns,
        "params": best.params,
        "val_metrics": best.val_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    print("Loading dataset...")
    df = clean_laptops_dataframe(load_raw_dataset())
    splits = split_dataset(df)
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
    print(f"  rows clean={len(df)}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    print("\nRunning experiments...")
    results = run_all(train_df, val_df)

    table = results_to_table(results)
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(EXPERIMENTS_CSV, index=False)
    print(f"\nExperiments table -> {EXPERIMENTS_CSV}")
    print(table.to_string(index=False))

    final = select_and_finalize(results, train_df, val_df, test_df)

    summary = {
        "seed": SEED,
        "rows_clean": len(df),
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "experiments": [
            {
                "name": r.name,
                "hypothesis": r.hypothesis,
                "n_features": len(r.feature_columns),
                "params": r.params,
                "val_metrics": r.val_metrics,
                "fit_seconds": r.fit_seconds,
            }
            for r in results
        ],
        "final_model": final,
    }
    EXPERIMENTS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMetrics summary -> {EXPERIMENTS_JSON}")


if __name__ == "__main__":
    main()

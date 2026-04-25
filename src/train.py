"""Train and save both models:
  - models/baseline_ridge.joblib   — Ridge with feature engineering
  - models/linreg_raw.joblib       — LinearRegression on raw features only
"""

from __future__ import annotations

import json

from src.modeling import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_LINREG_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_baseline_pipeline,
    build_linreg_raw_pipeline,
    evaluate_regression,
    save_model,
    split_dataset,
)
from src.preprocessing import clean_laptops_dataframe, load_raw_dataset

METRICS_PATH = DEFAULT_MODEL_PATH.parent / "baseline_metrics.json"


def main() -> None:
    print("Loading dataset...")
    raw = load_raw_dataset()
    df = clean_laptops_dataframe(raw)
    print(f"  Clean rows: {len(df)}")

    splits = split_dataset(df)
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ── Ridge with feature engineering ──────────────────────────────────────
    print("\nTraining Ridge (all features)...")
    ridge_features = [c for c in BASE_FEATURE_COLUMNS if c in train_df.columns]
    ridge = build_baseline_pipeline(train_df)
    ridge.fit(train_df[ridge_features], train_df[TARGET_COLUMN])

    val_metrics_ridge = evaluate_regression(
        val_df[TARGET_COLUMN],
        ridge.predict(val_df[ridge_features]),
    )
    print(f"  Validation → MAE={val_metrics_ridge['mae']:.0f}  R²={val_metrics_ridge['r2']:.3f}")

    # Retrain on train+val before final evaluation
    train_val = splits["train"]._append(splits["validation"]).reset_index(drop=True)
    ridge_final = build_baseline_pipeline(train_val)
    ridge_final.fit(train_val[ridge_features], train_val[TARGET_COLUMN])
    test_metrics_ridge = evaluate_regression(
        test_df[TARGET_COLUMN],
        ridge_final.predict(test_df[ridge_features]),
    )
    print(f"  Test      → MAE={test_metrics_ridge['mae']:.0f}  R²={test_metrics_ridge['r2']:.3f}")
    save_model(ridge_final, DEFAULT_MODEL_PATH)
    print(f"  Saved → {DEFAULT_MODEL_PATH}")

    # ── LinearRegression without feature engineering ─────────────────────────
    print("\nTraining LinearRegression (raw features only)...")
    raw_features = [c for c in RAW_FEATURE_COLUMNS if c in train_df.columns]
    linreg = build_linreg_raw_pipeline()
    linreg.fit(train_df[raw_features], train_df[TARGET_COLUMN])

    val_metrics_lr = evaluate_regression(
        val_df[TARGET_COLUMN],
        linreg.predict(val_df[raw_features]),
    )
    print(f"  Validation → MAE={val_metrics_lr['mae']:.0f}  R²={val_metrics_lr['r2']:.3f}")

    linreg_final = build_linreg_raw_pipeline()
    linreg_final.fit(train_val[raw_features], train_val[TARGET_COLUMN])
    test_metrics_lr = evaluate_regression(
        test_df[TARGET_COLUMN],
        linreg_final.predict(test_df[raw_features]),
    )
    print(f"  Test      → MAE={test_metrics_lr['mae']:.0f}  R²={test_metrics_lr['r2']:.3f}")
    save_model(linreg_final, DEFAULT_LINREG_MODEL_PATH)
    print(f"  Saved → {DEFAULT_LINREG_MODEL_PATH}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "seed": 42,
        "ridge": {
            "validation": val_metrics_ridge,
            "test": test_metrics_ridge,
        },
        "linreg_raw": {
            "validation": val_metrics_lr,
            "test": test_metrics_lr,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nMetrics saved → {METRICS_PATH}")


if __name__ == "__main__":
    main()

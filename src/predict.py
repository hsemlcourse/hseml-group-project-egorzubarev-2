"""Predict laptop price from CLI arguments.

Example:
    python -m src.predict \\
        --brand Lenovo --ram_gb 16 --storage_gb 512 \\
        --screen_diagonal_inch 15.6 --cpu "Intel Core i5-12450H" \\
        --gpu "NVIDIA GeForce RTX 3050" --os "No OS" --model ridge

Available --model values:
    ridge      — Ridge with feature engineering  (default)
    linreg     — LinearRegression raw features only
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.modeling import DEFAULT_LINREG_MODEL_PATH, DEFAULT_MODEL_PATH
from src.preprocessing import add_engineered_features

MODEL_PATHS = {
    "ridge": DEFAULT_MODEL_PATH,
    "linreg": DEFAULT_LINREG_MODEL_PATH,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Предсказать цену ноутбука по его характеристикам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--brand", default=None, help="Бренд, напр. Lenovo")
    parser.add_argument("--screen_diagonal_inch", type=float, default=None, help="Диагональ, дюймы")
    parser.add_argument("--screen_resolution", default=None, help="Разрешение, напр. 1920x1080")
    parser.add_argument("--matrix_type", default=None, help="Тип матрицы, напр. IPS")
    parser.add_argument("--cpu", default=None, help="Процессор, напр. 'Intel Core i5-12450H'")
    parser.add_argument("--gpu", default=None, help="Видеокарта, напр. 'NVIDIA GeForce RTX 3050'")
    parser.add_argument("--ram_gb", type=float, default=None, help="ОЗУ, ГБ")
    parser.add_argument("--ram_type", default=None, help="Тип ОЗУ, напр. DDR5")
    parser.add_argument("--storage_gb", type=float, default=None, help="Накопитель, ГБ")
    parser.add_argument("--os", default=None, help="ОС, напр. 'No OS' или 'Windows 11'")
    parser.add_argument("--weight_kg", type=float, default=None, help="Вес, кг")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        default="ridge",
        help="Какую модель использовать: ridge (по умолчанию) или linreg",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_path = MODEL_PATHS[args.model]
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        print("Сначала обучите модели: python -m src.train")
        raise SystemExit(1)

    model = joblib.load(model_path)

    row = {
        "brand": args.brand,
        "screen_diagonal_inch": args.screen_diagonal_inch,
        "screen_resolution": args.screen_resolution,
        "matrix_type": args.matrix_type,
        "cpu": args.cpu,
        "gpu": args.gpu,
        "ram_gb": args.ram_gb,
        "ram_type": args.ram_type,
        "storage_gb": args.storage_gb,
        "os": args.os,
        "weight_kg": args.weight_kg,
    }

    df = pd.DataFrame([row])

    # Ridge uses engineered features — add them
    if args.model == "ridge":
        df = add_engineered_features(df)

    price = model.predict(df)[0]
    print(f"\nПредсказанная цена: {price:,.0f} руб.")

    # Show which fields were provided
    provided = {k: v for k, v in row.items() if v is not None}
    missing = [k for k, v in row.items() if v is None]
    print(f"\nПредоставлено признаков: {list(provided.keys())}")
    if missing:
        print(f"Пропущено (будут заполнены медианой/модой): {missing}")


if __name__ == "__main__":
    main()

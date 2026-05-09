from __future__ import annotations

import json

import pandas as pd

from src.project_config import CSV_PATH, MAX_REASONABLE_PRICE_RUB

JSON_COLUMNS = ["sources", "source_urls", "raw_specs_json"]


def _safe_json_loads(value: object) -> object:
    if not isinstance(value, str) or not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_raw_dataset(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path or str(CSV_PATH)
    dataframe = pd.read_csv(path)
    for column in JSON_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(_safe_json_loads)
    return dataframe


def _extract_resolution_parts(resolution: pd.Series) -> tuple[pd.Series, pd.Series]:
    extracted = resolution.fillna("").str.extract(r"(?P<width>\d{3,4})x(?P<height>\d{3,4})")
    width = pd.to_numeric(extracted["width"], errors="coerce")
    height = pd.to_numeric(extracted["height"], errors="coerce")
    return width, height


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    resolution_series = (
        result["screen_resolution"]
        if "screen_resolution" in result.columns
        else pd.Series("", index=result.index, dtype=object)
    )
    width, height = _extract_resolution_parts(resolution_series)
    result["resolution_width"] = width
    result["resolution_height"] = height
    result["screen_pixels_mln"] = (width * height) / 1_000_000
    result["cpu_vendor"] = (
        result["cpu"].fillna("Unknown").str.extract(r"^(Intel|AMD|Apple|Qualcomm)", expand=False).fillna("Other")
    )
    result["gpu_vendor"] = (
        result["gpu"].fillna("Unknown").str.extract(r"^(NVIDIA|Intel|AMD|Apple|Qualcomm)", expand=False).fillna("Other")
    )
    result["is_discrete_gpu"] = (
        result["gpu"].fillna("").str.contains(r"GeForce|RTX|Radeon RX|Arc", case=False, regex=True).astype(int)
    )
    result["is_without_os"] = result["os"].fillna("").isin(["No OS", "FreeDOS", "DOS"]).astype(int)
    result["is_gaming_series"] = (
        result.get("name", pd.Series("", index=result.index))
        .fillna("")
        .str.contains(
            r"gaming|rog|tuf|predator|legion|cyborg|katana|raider|victus|nitro",
            case=False,
            regex=True,
        )
        .astype(int)
    )
    return result


def clean_laptops_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    numeric_columns = [
        "price_rub",
        "screen_diagonal_inch",
        "ram_gb",
        "storage_gb",
        "weight_kg",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "brand" in result.columns:
        result["brand"] = result["brand"].fillna("Unknown")
    if "matrix_type" in result.columns:
        result["matrix_type"] = result["matrix_type"].fillna("Unknown")
    if "os" in result.columns:
        result["os"] = result["os"].fillna("Unknown")
    if "name" in result.columns:
        result["name"] = result["name"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    if "price_rub" in result.columns:
        result = result[result["price_rub"].le(MAX_REASONABLE_PRICE_RUB) | result["price_rub"].isna()]

    result = result.dropna(subset=["price_rub"]).drop_duplicates(subset=["fingerprint"]).reset_index(drop=True)
    result = add_engineered_features(result)
    return result

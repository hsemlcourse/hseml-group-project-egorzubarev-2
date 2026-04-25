from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from typing import Any

import pandas as pd

from src.http_utils import compact_json_ready, median_int
from src.project_config import CSV_PATH, DB_PATH, RAW_DIR
from src.scraper_common import LaptopRecord


def _canonical_title(title: str) -> str:
    cleaned = title.lower()
    cleaned = cleaned.replace("ноутбук игровой", "")
    cleaned = cleaned.replace("ноутбук", "")
    cleaned = cleaned.replace("ультрабук", "")
    cleaned = cleaned.strip()
    return " ".join(cleaned.split())


def build_fingerprint(record: LaptopRecord) -> str:
    payload = {
        "title": _canonical_title(record.title),
        "brand": record.brand,
        "screen_diagonal_inch": record.screen_diagonal_inch,
        "screen_resolution": record.screen_resolution,
        "matrix_type": record.matrix_type,
        "cpu": record.cpu,
        "gpu": record.gpu,
        "ram_gb": record.ram_gb,
        "ram_type": record.ram_type,
        "storage_gb": record.storage_gb,
        "os": record.os,
        "weight_kg": record.weight_kg,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def merge_duplicates(records: list[LaptopRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[LaptopRecord]] = defaultdict(list)
    for record in records:
        grouped[build_fingerprint(record)].append(record)

    merged_records: list[dict[str, Any]] = []
    for fingerprint, items in grouped.items():
        prices = [item.price_rub for item in items if item.price_rub is not None]
        reference = items[0]
        raw_specs = {item.source: compact_json_ready(item.raw_specs) for item in items}
        merged_records.append(
            {
                "fingerprint": fingerprint,
                "name": reference.title,
                "brand": reference.brand,
                "price_rub": median_int(prices),
                "screen_diagonal_inch": reference.screen_diagonal_inch,
                "screen_resolution": reference.screen_resolution,
                "matrix_type": reference.matrix_type,
                "cpu": reference.cpu,
                "gpu": reference.gpu,
                "ram_gb": reference.ram_gb,
                "ram_type": reference.ram_type,
                "storage_gb": reference.storage_gb,
                "os": reference.os,
                "weight_kg": reference.weight_kg,
                "sources": json.dumps(sorted({item.source for item in items}), ensure_ascii=False),
                "source_urls": json.dumps({item.source: item.product_url for item in items}, ensure_ascii=False),
                "raw_specs_json": json.dumps(raw_specs, ensure_ascii=False),
            }
        )
    return merged_records


def save_dataset(records: list[dict[str, Any]]) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records).sort_values(["price_rub", "name"], na_position="last").reset_index(drop=True)
    dataframe.to_csv(CSV_PATH, index=False)

    connection = sqlite3.connect(DB_PATH)
    dataframe.to_sql("laptops", connection, if_exists="replace", index=False)
    connection.close()
    return dataframe

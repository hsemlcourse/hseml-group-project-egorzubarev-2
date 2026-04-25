from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.citilink_scraper import extract_citilink_product_id
from src.dns_scraper import build_dns_record_from_snapshot
from src.project_config import CSV_PATH, RAW_DIR
from src.scraper_common import LaptopRecord
from src.storage import merge_duplicates, save_dataset

BROWSER_BATCH_DIR = RAW_DIR / "browser_batches"


def _json_load(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_citilink_records(csv_path: Path = CSV_PATH) -> list[LaptopRecord]:
    if not csv_path.exists():
        return []

    dataframe = pd.read_csv(csv_path)
    records: list[LaptopRecord] = []
    for row in dataframe.to_dict(orient="records"):
        sources = _json_load(row.get("sources", "[]"))
        if isinstance(sources, list) and "citilink" not in sources:
            continue

        source_urls = _json_load(row.get("source_urls", "{}")) or {}
        raw_specs_json = _json_load(row.get("raw_specs_json", "{}")) or {}
        product_url = source_urls.get("citilink", "") if isinstance(source_urls, dict) else ""
        raw_specs = raw_specs_json.get("citilink", {}) if isinstance(raw_specs_json, dict) else {}

        records.append(
            LaptopRecord(
                source="citilink",
                source_id=extract_citilink_product_id(product_url) if product_url else row["fingerprint"],
                product_url=product_url,
                title=row["name"],
                price_rub=int(row["price_rub"]) if pd.notna(row["price_rub"]) else None,
                brand=row.get("brand"),
                screen_diagonal_inch=(
                    float(row["screen_diagonal_inch"]) if pd.notna(row["screen_diagonal_inch"]) else None
                ),
                screen_resolution=row.get("screen_resolution"),
                matrix_type=row.get("matrix_type"),
                cpu=row.get("cpu"),
                gpu=row.get("gpu"),
                ram_gb=int(row["ram_gb"]) if pd.notna(row["ram_gb"]) else None,
                ram_type=row.get("ram_type"),
                storage_gb=int(row["storage_gb"]) if pd.notna(row["storage_gb"]) else None,
                os=row.get("os"),
                weight_kg=float(row["weight_kg"]) if pd.notna(row["weight_kg"]) else None,
                raw_specs=raw_specs,
            )
        )
    return records


def load_dns_records(batch_dir: Path = BROWSER_BATCH_DIR) -> list[LaptopRecord]:
    records: list[LaptopRecord] = []
    for path in sorted(batch_dir.glob("dns_products_repair_*.json")):
        snapshots = json.loads(path.read_text(encoding="utf-8"))
        for snapshot in snapshots:
            if "product" not in snapshot:
                continue
            records.append(build_dns_record_from_snapshot(snapshot))
    return records


def finalize_dataset() -> pd.DataFrame:
    records = [*load_citilink_records(), *load_dns_records()]
    merged_records = merge_duplicates(records)
    return save_dataset(merged_records)


def main() -> None:
    dataframe = finalize_dataset()
    print(f"Saved combined dataset with {len(dataframe)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
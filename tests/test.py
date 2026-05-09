from __future__ import annotations

import pandas as pd
from src.citilink_scraper import build_citilink_record
from src.dns_scraper import _build_dns_record_from_flattened
from src.modeling import evaluate_regression
from src.preprocessing import clean_laptops_dataframe
from src.scraper_common import LaptopRecord, clean_text, parse_core_features
from src.storage import merge_duplicates


def test_parse_core_features_extracts_main_laptop_fields() -> None:
    text = (
        'Ноутбук Lenovo IdeaPad Slim 3 15AMN8, 15.6", 2025, IPS, AMD Ryzen 5 7520U, '
        "16ГБ LPDDR5, 1ТБ SSD, AMD Radeon 610M, без операционной системы"
    )
    features = parse_core_features(text)

    assert features["brand"] == "Lenovo"
    assert features["screen_diagonal_inch"] == 15.6
    assert features["screen_resolution"] is None
    assert features["matrix_type"] == "IPS"
    assert features["cpu"] == "AMD Ryzen 5 7520U"
    assert features["ram_gb"] == 16
    assert features["storage_gb"] == 1024
    assert features["os"] == "No OS"


def test_merge_duplicates_collapses_identical_records() -> None:
    base_record = LaptopRecord(
        source="citilink",
        source_id="1",
        product_url="https://example.com/1",
        title="Ноутбук Lenovo IdeaPad Slim 3 15AMN8",
        price_rub=50000,
        brand="Lenovo",
        screen_diagonal_inch=15.6,
        screen_resolution="1920x1080",
        matrix_type="IPS",
        cpu="AMD Ryzen 5 7520U",
        gpu="AMD Radeon 610M",
        ram_gb=16,
        ram_type="LPDDR5",
        storage_gb=1024,
        os="No OS",
        weight_kg=1.6,
    )
    second_record = LaptopRecord(
        source="dns",
        source_id="2",
        product_url="https://example.com/2",
        title="Ноутбук Lenovo IdeaPad Slim 3 15AMN8",
        price_rub=52000,
        brand="Lenovo",
        screen_diagonal_inch=15.6,
        screen_resolution="1920x1080",
        matrix_type="IPS",
        cpu="AMD Ryzen 5 7520U",
        gpu="AMD Radeon 610M",
        ram_gb=16,
        ram_type="LPDDR5",
        storage_gb=1024,
        os="No OS",
        weight_kg=1.6,
    )

    merged = merge_duplicates([base_record, second_record])

    assert len(merged) == 1
    assert merged[0]["price_rub"] == 51000


def test_clean_laptops_dataframe_adds_engineered_columns() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "fingerprint": "abc",
                "name": "Ноутбук MSI Cyborg 15",
                "brand": "Msi",
                "price_rub": 99990,
                "screen_diagonal_inch": 15.6,
                "screen_resolution": "1920x1080",
                "matrix_type": "IPS",
                "cpu": "Intel Core i5-12450H",
                "gpu": "NVIDIA GeForce RTX 3050",
                "ram_gb": 16,
                "ram_type": "DDR5",
                "storage_gb": 512,
                "os": "No OS",
                "weight_kg": 1.8,
            }
        ]
    )

    cleaned = clean_laptops_dataframe(dataframe)

    assert cleaned.loc[0, "resolution_width"] == 1920
    assert cleaned.loc[0, "resolution_height"] == 1080
    assert cleaned.loc[0, "cpu_vendor"] == "Intel"
    assert cleaned.loc[0, "gpu_vendor"] == "NVIDIA"
    assert cleaned.loc[0, "is_gaming_series"] == 1


def test_clean_laptops_dataframe_drops_unreasonable_price_outliers() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "fingerprint": "valid",
                "name": "Ноутбук Lenovo IdeaPad Slim 3",
                "brand": "Lenovo",
                "price_rub": 89990,
                "screen_resolution": "1920x1080",
                "cpu": "AMD Ryzen 5 7520U",
                "gpu": "AMD Radeon 610M",
                "os": "No OS",
            },
            {
                "fingerprint": "broken",
                "name": "Ноутбук Lenovo IdeaPad Slim 3",
                "brand": "Lenovo",
                "price_rub": 9_994_535_4390,
                "screen_resolution": "1920x1080",
                "cpu": "AMD Ryzen 5 7520U",
                "gpu": "AMD Radeon 610M",
                "os": "No OS",
            },
        ]
    )

    cleaned = clean_laptops_dataframe(dataframe)

    assert cleaned["fingerprint"].tolist() == ["valid"]


def test_build_citilink_record_prefers_catalog_price() -> None:
    record = build_citilink_record(
        {
            "href": "/product/test-123/",
            "title": "Ноутбук Lenovo IdeaPad Slim 3 15AMN8",
            "priceText": "59 990 ₽",
        },
        {
            "title": "Ноутбук Lenovo IdeaPad Slim 3 15AMN8",
            "price_rub": 9_994_535_4390,
            "features": {
                "brand": "Lenovo",
                "screen_diagonal_inch": 15.6,
                "screen_resolution": "1920x1080",
                "matrix_type": "IPS",
                "cpu": "AMD Ryzen 5 7520U",
                "gpu": "AMD Radeon 610M",
                "ram_gb": 16,
                "ram_type": "LPDDR5",
                "storage_gb": 512,
                "os": "No OS",
                "weight_kg": 1.62,
            },
            "raw_specs": {},
        },
    )

    assert record.price_rub == 59_990


def test_evaluate_regression_returns_expected_metrics() -> None:
    metrics = evaluate_regression(pd.Series([100.0, 200.0]), [110.0, 190.0])

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == 10.0


def test_clean_text_handles_none_and_non_strings() -> None:
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("  Lenovo\xa0IdeaPad   Slim  ") == "Lenovo IdeaPad Slim"


def test_dns_record_returns_none_when_price_is_missing() -> None:
    card = {"guid": "abc", "href": "/product/x/", "title": "Ноутбук X", "priceText": ""}
    product = {"guid": "abc", "name": "Ноутбук X", "specs": "", "description": "", "price": None}
    record = _build_dns_record_from_flattened(card=card, product=product, flattened={})

    assert record.price_rub is None


def test_dns_record_falls_back_to_card_price_text() -> None:
    card = {"guid": "abc", "href": "/product/x/", "title": "Ноутбук X", "priceText": "59 990 ₽"}
    product = {"guid": "abc", "name": "Ноутбук X", "specs": "", "description": "", "price": 0}
    record = _build_dns_record_from_flattened(card=card, product=product, flattened={})

    assert record.price_rub == 59_990

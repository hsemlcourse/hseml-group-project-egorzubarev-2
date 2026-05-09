from __future__ import annotations

from typing import Any

from src.http_utils import make_absolute_url
from src.scraper_common import LaptopRecord, merge_text_features, parse_price_rub


def flatten_dns_characteristics(characteristics: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for section, entries in characteristics.items():
        for entry in entries:
            title = str(entry.get("title", "")).strip()
            value = str(entry.get("value", "")).strip()
            if not title or not value:
                continue
            key = f"{section} / {title}"
            flattened[key] = value
    return flattened


def _resolve_dns_price(product: dict[str, Any], card: dict[str, Any]) -> int | None:
    raw_price = product.get("price")
    if raw_price not in (None, "", 0):
        try:
            return int(raw_price)
        except (TypeError, ValueError):
            pass
    return parse_price_rub(card.get("priceText", ""))


def _build_dns_record_from_flattened(
    card: dict[str, Any],
    product: dict[str, Any],
    flattened: dict[str, str],
) -> LaptopRecord:
    feature_text = " ".join(
        [
            product.get("name", ""),
            product.get("specs", ""),
            product.get("description", ""),
            " ".join(f"{key}: {value}" for key, value in flattened.items()),
        ]
    )
    features = merge_text_features(feature_text)
    return LaptopRecord(
        source="dns",
        source_id=str(product.get("guid") or card["guid"]),
        product_url=make_absolute_url("https://www.dns-shop.ru", card["href"]),
        title=product.get("name") or card["title"],
        price_rub=_resolve_dns_price(product, card),
        brand=features.get("brand"),
        screen_diagonal_inch=features.get("screen_diagonal_inch"),
        screen_resolution=features.get("screen_resolution"),
        matrix_type=features.get("matrix_type"),
        cpu=features.get("cpu"),
        gpu=features.get("gpu"),
        ram_gb=features.get("ram_gb"),
        ram_type=features.get("ram_type"),
        storage_gb=features.get("storage_gb"),
        os=features.get("os"),
        weight_kg=features.get("weight_kg"),
        raw_specs=flattened | {"specs": product.get("specs", ""), "description": product.get("description", "")},
    )


def build_dns_record(card: dict[str, Any], payload: dict[str, Any]) -> LaptopRecord:
    data = payload["data"]
    flattened = flatten_dns_characteristics(data.get("characteristics", {}))
    return _build_dns_record_from_flattened(card=card, product=data, flattened=flattened)


def build_dns_record_from_snapshot(snapshot: dict[str, Any]) -> LaptopRecord:
    return _build_dns_record_from_flattened(
        card=snapshot["card"],
        product=snapshot["product"],
        flattened=snapshot["product"].get("characteristics", {}),
    )

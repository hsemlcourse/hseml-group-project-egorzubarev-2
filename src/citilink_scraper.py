from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from src.http_utils import get_html, make_absolute_url
from src.project_config import MAX_REASONABLE_PRICE_RUB
from src.scraper_common import LaptopRecord, merge_text_features, parse_price_rub

SUMMARY_LABELS = {
    "Экран": "screen",
    "Процессор": "cpu",
    "Графический процессор": "gpu",
    "Оперативная память": "ram",
    "Диск": "storage",
    "Операционная система": "os",
    "Клавиатура": "keyboard",
    "Вес": "weight",
}


def extract_citilink_product_id(url: str) -> str:
    match = re.search(r"-(\d+)/?$", url)
    if not match:
        raise ValueError(f"Could not parse Citilink product id from URL: {url}")
    return match.group(1)


def _extract_labeled_pairs(strings: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for index, value in enumerate(strings[:-1]):
        label = value.rstrip(":")
        if label not in SUMMARY_LABELS:
            continue
        next_value = strings[index + 1]
        if next_value.rstrip(":") in SUMMARY_LABELS:
            continue
        pairs[label] = next_value
    return pairs


def _extract_description(text: str) -> str:
    match = re.search(r"Описание\s+(.*?)\s+Сообщить об ошибке в описании", text, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def fetch_citilink_details(session, product_url: str) -> dict[str, Any]:
    main_html = get_html(session, product_url)
    properties_html = get_html(session, f"{product_url.rstrip('/')}/properties/")

    main_soup = BeautifulSoup(main_html, "lxml")
    properties_soup = BeautifulSoup(properties_html, "lxml")

    main_strings = [value.strip() for value in main_soup.stripped_strings if value.strip()]
    properties_strings = [value.strip() for value in properties_soup.stripped_strings if value.strip()]

    summary_pairs = _extract_labeled_pairs(main_strings)
    description = _extract_description(main_soup.get_text(" ", strip=True))
    properties_text = " ".join(properties_strings)
    feature_text = " ".join([*summary_pairs.values(), properties_text, description])
    features = merge_text_features(feature_text)

    weight_match = re.search(r"весит\s+(\d(?:[.,]\d{1,2})?)\s*кг", description, flags=re.IGNORECASE)
    if features.get("weight_kg") is None and weight_match:
        features["weight_kg"] = float(weight_match.group(1).replace(",", "."))

    extra_pairs = {}
    for label, value in summary_pairs.items():
        extra_pairs[label] = value
    if description:
        extra_pairs["Описание"] = description

    title_node = main_soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else ""
    price = parse_price_rub(main_soup.get_text(" ", strip=True))

    return {
        "title": title,
        "price_rub": price,
        "features": features,
        "raw_specs": extra_pairs,
    }


def build_citilink_record(card: dict[str, Any], details: dict[str, Any]) -> LaptopRecord:
    product_url = make_absolute_url("https://www.citilink.ru", card["href"])
    source_id = extract_citilink_product_id(product_url)
    title = details["title"] or card["title"]
    features = details["features"]
    catalog_price = parse_price_rub(card["priceText"])
    page_price = details["price_rub"]
    resolved_price = catalog_price
    if resolved_price is None and page_price is not None and page_price <= MAX_REASONABLE_PRICE_RUB:
        resolved_price = page_price

    return LaptopRecord(
        source="citilink",
        source_id=source_id,
        product_url=product_url,
        title=title,
        price_rub=resolved_price,
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
        raw_specs=details["raw_specs"],
    )

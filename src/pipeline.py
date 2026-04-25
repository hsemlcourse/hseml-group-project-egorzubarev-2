from __future__ import annotations

import argparse
from collections.abc import Iterable

from tqdm import tqdm

from src.browser_fetcher import BrowserCatalogClient
from src.citilink_scraper import build_citilink_record, fetch_citilink_details
from src.dns_scraper import build_dns_record
from src.http_utils import build_retry_session
from src.storage import merge_duplicates, save_dataset


def _unique_by_key(items: Iterable[dict], key: str) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        value = str(item[key])
        if value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result


def collect_citilink_cards(browser: BrowserCatalogClient, max_pages: int | None = None) -> list[dict]:
    cards: list[dict] = []
    seen_page_signatures: set[tuple[str, ...]] = set()
    page_number = 1
    progress = tqdm(desc="Citilink pages")
    while True:
        page_cards = browser.fetch_citilink_catalog_cards(page_number)
        signature = tuple(card["href"] for card in page_cards[:5])
        if not page_cards or signature in seen_page_signatures:
            break
        seen_page_signatures.add(signature)
        cards.extend(page_cards)
        progress.update(1)
        if max_pages is not None and page_number >= max_pages:
            break
        page_number += 1
    progress.close()
    return _unique_by_key(cards, "href")


def collect_dns_cards(browser: BrowserCatalogClient, max_pages: int | None = None) -> list[dict]:
    total_pages = browser.get_dns_total_pages()
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)
    cards: list[dict] = []
    for page_number in tqdm(range(1, total_pages + 1), desc="DNS pages"):
        cards.extend(browser.fetch_dns_catalog_cards(page_number))
    return _unique_by_key(cards, "guid")


def scrape_all(
    max_citilink_pages: int | None = None,
    max_dns_pages: int | None = None,
    include_citilink: bool = True,
    include_dns: bool = True,
    headless: bool = True,
) -> list[dict]:
    if not include_citilink and not include_dns:
        raise ValueError("At least one source must be enabled")

    session = build_retry_session()
    records = []
    with BrowserCatalogClient(headless=headless) as browser:
        if include_citilink:
            citilink_cards = collect_citilink_cards(browser, max_pages=max_citilink_pages)
            for card in tqdm(citilink_cards, desc="Citilink details"):
                details = fetch_citilink_details(session, f"https://www.citilink.ru{card['href']}")
                records.append(build_citilink_record(card, details))

        if include_dns:
            try:
                dns_cards = collect_dns_cards(browser, max_pages=max_dns_pages)
            except Exception as exc:  # pragma: no cover - depends on anti-bot state
                raise RuntimeError(
                    "DNS blocked the current browser session. Retry in an interactive session or use saved raw data."
                ) from exc
            for card in tqdm(dns_cards, desc="DNS details"):
                payload = browser.fetch_dns_product_payload(card["guid"])
                records.append(build_dns_record(card, payload))

    merged_records = merge_duplicates(records)
    save_dataset(merged_records)
    return merged_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape laptop data for CP1")
    parser.add_argument("--max-citilink-pages", type=int, default=None)
    parser.add_argument("--max-dns-pages", type=int, default=None)
    parser.add_argument("--skip-citilink", action="store_true")
    parser.add_argument("--skip-dns", action="store_true")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    scrape_all(
        max_citilink_pages=args.max_citilink_pages,
        max_dns_pages=args.max_dns_pages,
        include_citilink=not args.skip_citilink,
        include_dns=not args.skip_dns,
        headless=not args.headed,
    )


if __name__ == "__main__":
    main()

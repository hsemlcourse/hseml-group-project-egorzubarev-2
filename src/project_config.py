from __future__ import annotations

from pathlib import Path

SEED = 42
REQUEST_TIMEOUT_SECONDS = 30
MIN_REQUEST_DELAY_SECONDS = 0.35
MAX_REQUEST_DELAY_SECONDS = 0.8
MAX_RETRIES = 4
MAX_REASONABLE_PRICE_RUB = 1_000_000

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = ROOT_DIR / "laptops.db"
CSV_PATH = RAW_DIR / "laptops.csv"

CITILINK_CATALOG_URL = "https://www.citilink.ru/catalog/noutbuki/"
DNS_CATALOG_URL = "https://www.dns-shop.ru/catalog/17a892f816404e77/noutbuki/"

USER_AGENTS = [
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"),
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    ),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"),
]

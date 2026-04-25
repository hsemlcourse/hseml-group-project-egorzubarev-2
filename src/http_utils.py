from __future__ import annotations

import random
import time
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.project_config import (
    MAX_REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    MIN_REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENTS,
)


def build_retry_session() -> requests.Session:
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=0.6,
        allowed_methods=["GET"],
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_headers() -> dict[str, str]:
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": random.choice(USER_AGENTS),
    }


def polite_sleep() -> None:
    time.sleep(random.uniform(MIN_REQUEST_DELAY_SECONDS, MAX_REQUEST_DELAY_SECONDS))


def get_html(session: requests.Session, url: str) -> str:
    polite_sleep()
    response = session.get(url, headers=build_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response.text


def make_absolute_url(base_url: str, url: str) -> str:
    return urljoin(base_url, url)


def median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return round((ordered[middle - 1] + ordered[middle]) / 2)


def compact_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): compact_json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [compact_json_ready(item) for item in value]
    return value

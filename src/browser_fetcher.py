from __future__ import annotations

import json
import re
from typing import Any

from src.http_utils import polite_sleep
from src.project_config import CITILINK_CATALOG_URL, DNS_CATALOG_URL, USER_AGENTS


class BrowserCatalogClient:
    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self.page = None

    def __enter__(self) -> BrowserCatalogClient:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for catalog scraping. "
                "Install the package and run 'playwright install chromium'."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=USER_AGENTS[0],
            locale="ru-RU",
            viewport={"width": 1440, "height": 1600},
        )
        self.page = self._context.new_page()
        self.page.set_extra_http_headers({"Accept-Language": "ru-RU,ru;q=0.9"})
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    def get_dns_total_pages(self) -> int:
        self.page.goto(DNS_CATALOG_URL, wait_until="domcontentloaded")
        title = self.page.title()
        match = re.search(r"из\s+(\d+)", title)
        if not match:
            raise RuntimeError("Could not determine DNS page count from page title")
        return int(match.group(1))

    def get_citilink_total_pages(self) -> int:
        self.page.goto(CITILINK_CATALOG_URL, wait_until="domcontentloaded")
        self.page.wait_for_function(
            """
            () => [...document.querySelectorAll('a[href*="/product/"]')]
              .filter(anchor => (anchor.textContent || '').includes('Ноутбук')).length >= 40
            """,
            timeout=20000,
        )
        total_pages = self.page.evaluate(
            """
            () => {
              const pageLinks = [...document.querySelectorAll('a[href*="?p="]')]
                .map(link => {
                  const href = link.getAttribute('href') || '';
                  const match = href.match(/[?&]p=(\d+)/);
                  return match ? Number(match[1]) : null;
                })
                .filter(Boolean);
              return pageLinks.length ? Math.max(...pageLinks) : null;
            }
            """
        )
        if not total_pages:
            raise RuntimeError("Could not determine Citilink page count from pagination links")
        return int(total_pages)

    def fetch_dns_catalog_cards(self, page_number: int) -> list[dict[str, Any]]:
        url = DNS_CATALOG_URL if page_number == 1 else f"{DNS_CATALOG_URL}?p={page_number}"
        self.page.goto(url, wait_until="domcontentloaded")
        polite_sleep()
        cards = self.page.evaluate(
            """
            () => [...document.querySelectorAll('.catalog-product[data-product]')]
              .map(card => {
                const links = [...card.querySelectorAll('a[href*="/product/"]')];
                const link = links.find(node => {
                  const candidate = (node.getAttribute('title') || node.textContent || '').trim();
                  return candidate.length > 0;
                }) || links[0] || null;
                const candidates = [...card.querySelectorAll('button, div, span')]
                  .map(node => (node.textContent || '').trim())
                  .filter(text => /₽/.test(text));
                return {
                  guid: card.dataset.product || '',
                  code: card.dataset.code || '',
                  href: link ? link.getAttribute('href') || '' : '',
                  title: link ? ((link.getAttribute('title') || link.textContent || '').trim()) : '',
                  priceText: candidates[0] || '',
                };
              })
              .filter(card => card.guid && card.href && card.title);
            """
        )
        return list(cards)

    def fetch_citilink_catalog_cards(self, page_number: int) -> list[dict[str, Any]]:
        url = CITILINK_CATALOG_URL if page_number == 1 else f"{CITILINK_CATALOG_URL}?p={page_number}"
        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_function(
            """
            () => [...document.querySelectorAll('a[href*="/product/"]')]
              .filter(anchor => (anchor.textContent || '').includes('Ноутбук')).length >= 40
            """,
            timeout=20000,
        )
        polite_sleep()
        cards = self.page.evaluate(
            """
            () => {
              const anchors = [...document.querySelectorAll('a[href*="/product/"]')]
                .filter(anchor => (anchor.textContent || '').includes('Ноутбук'));
              const cards = [];
              const seen = new Set();
              for (const anchor of anchors) {
                const href = anchor.getAttribute('href') || '';
                if (!href || seen.has(href)) {
                  continue;
                }
                let node = anchor;
                let priceText = '';
                for (let depth = 0; depth < 6 && node; depth += 1) {
                  node = node.parentElement;
                  if (!node) {
                    break;
                  }
                  const buttons = [...node.querySelectorAll('button')].map(button => (button.textContent || '').trim());
                  priceText = buttons.find(text => /₽/.test(text)) || priceText;
                  if (priceText) {
                    break;
                  }
                }
                if (!priceText) {
                  continue;
                }
                seen.add(href);
                cards.push({
                  href,
                  title: (anchor.textContent || '').trim(),
                  priceText,
                });
              }
              return cards;
            }
            """
        )
        return list(cards)

    def fetch_dns_product_payload(self, guid: str) -> dict[str, Any]:
        endpoint = f"/pwa/pwa/get-product/?id={guid}"
        raw_payload = self.page.evaluate(
            """
            async endpoint => {
              const response = await fetch(endpoint);
              if (!response.ok) {
                throw new Error(`DNS product endpoint failed: ${response.status}`);
              }
              return await response.text();
            }
            """,
            endpoint,
        )
        polite_sleep()
        return json.loads(raw_payload)

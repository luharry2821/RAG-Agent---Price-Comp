"""Source adapters: one per retailer, plus the shared normalization step."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, quote_plus

from ..models import Listing, utcnow
from ..normalize import (
    detect_gender,
    normalize_brand,
    normalize_colorway,
    normalize_sizes,
    normalize_style_code,
    parse_price,
    parse_title,
)
from .http import FetchError, HttpClient
from .jsonld import extract_listing_fields


@dataclass(frozen=True)
class SiteSpec:
    """Static description of a retailer we can pull listings from."""

    key: str
    name: str
    home: str
    search_url: str                       # {q} is replaced with the url-encoded query
    product_link: str                     # regex matching product page hrefs
    brands: tuple[str, ...] = ()          # () = carries all three brands
    notes: str = ""

    def sells(self, brand: str) -> bool:
        return not self.brands or brand in self.brands


def build_listing(spec: SiteSpec, data: dict[str, Any]) -> Listing | None:
    """Turn raw per-site fields into a normalized :class:`Listing`.

    Every adapter funnels through here so that brand/model/colorway/style-code
    normalization is identical no matter where the data came from.
    """
    title = (data.get("title") or "").strip()
    url = data.get("url") or ""
    if not title or not url:
        return None

    price, currency = parse_price(data.get("price"))
    if price is None:
        return None
    list_price, _ = parse_price(data.get("list_price"))

    parsed = parse_title(title, brand_hint=str(data.get("brand") or ""))
    brand = normalize_brand(str(data.get("brand") or "")) or parsed["brand"]
    if not brand and len(spec.brands) == 1:
        brand = spec.brands[0]      # single-brand store: the site itself tells us
    style_code = normalize_style_code(str(data.get("style_code") or "")) or parsed["style_code"]
    colorway = normalize_colorway(str(data.get("colorway") or "")) or parsed["colorway"]
    gender = (str(data.get("gender") or "").strip().lower()
              or parsed["gender"] or detect_gender(title))
    model = data.get("model") or parsed["model"]

    shipping, _ = parse_price(data.get("shipping"))

    return Listing(
        source=spec.key,
        source_name=spec.name,
        url=urljoin(spec.home, url),
        title=title,
        brand=brand,
        model=model,
        colorway=colorway,
        style_code=style_code,
        retailer_sku=str(data.get("retailer_sku") or ""),
        price=price,
        list_price=list_price,
        currency=(data.get("currency") or currency or "USD").upper(),
        shipping=shipping,
        in_stock=bool(data.get("in_stock", True)),
        sizes=normalize_sizes(data.get("sizes")),
        gender=gender if gender in ("men", "women", "kids", "unisex") else "",
        condition=str(data.get("condition") or "new"),
        image=str(data.get("image") or ""),
        scraped_at=str(data.get("scraped_at") or utcnow()),
        raw={k: v for k, v in data.items() if k not in ("raw",)},
    )


class SourceAdapter:
    """Interface every retailer adapter implements."""

    spec: SiteSpec

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def name(self) -> str:
        return self.spec.name

    def search(self, query: str, limit: int = 20) -> list[Listing]:
        raise NotImplementedError

    def collect(self, queries: Sequence[str], limit: int = 20) -> list[Listing]:
        seen: dict[str, Listing] = {}
        for q in queries:
            for listing in self.search(q, limit=limit):
                seen.setdefault(listing.listing_id, listing)
        return list(seen.values())


class LiveSource(SourceAdapter):
    """Fetches real pages: search results page -> product pages -> JSON-LD.

    Live mode is opt-in (``--live`` / ``SNEAKERRAG_MODE=live``). Before turning
    it on for a site, check that site's Terms of Service and robots.txt — many
    retailers require an affiliate or partner API instead of crawling, and this
    client will refuse disallowed paths.
    """

    def __init__(self, spec: SiteSpec, client: HttpClient | None = None) -> None:
        self.spec = spec
        self.client = client or HttpClient()

    def search_url(self, query: str) -> str:
        return self.spec.search_url.format(q=quote_plus(query))

    def product_urls(self, html: str, limit: int) -> list[str]:
        pattern = re.compile(self.spec.product_link, re.I)
        urls: list[str] = []
        for match in re.finditer(r'href=["\']([^"\']+)["\']', html or "", re.I):
            href = match.group(1)
            if pattern.search(href):
                full = urljoin(self.spec.home, href.split("#")[0])
                if full not in urls:
                    urls.append(full)
            if len(urls) >= limit:
                break
        return urls

    def search(self, query: str, limit: int = 20) -> list[Listing]:
        try:
            html = self.client.get(self.search_url(query))
        except FetchError:
            return []
        listings: list[Listing] = []
        for url in self.product_urls(html, limit):
            try:
                page = self.client.get(url)
            except FetchError:
                continue
            data = extract_listing_fields(page)
            data.setdefault("url", url)
            if not data.get("url"):
                data["url"] = url
            listing = build_listing(self.spec, data)
            if listing:
                listings.append(listing)
        return listings


class StaticSource(SourceAdapter):
    """Adapter backed by records already in memory (fixtures, exports, APIs)."""

    def __init__(self, spec: SiteSpec, records: Iterable[dict[str, Any]]) -> None:
        self.spec = spec
        self._listings: list[Listing] = []
        for record in records:
            listing = build_listing(spec, record)
            if listing:
                self._listings.append(listing)

    def all(self) -> list[Listing]:
        return list(self._listings)

    def search(self, query: str, limit: int = 20) -> list[Listing]:
        if not query:
            return self.all()[:limit]
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        scored = []
        for listing in self._listings:
            haystack = listing.summary().lower()
            score = sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, listing))
        scored.sort(key=lambda s: -s[0])
        return [l for _, l in scored[:limit]]

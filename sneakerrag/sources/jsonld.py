"""Extract product/offer data from schema.org markup.

Most large retailers embed a ``Product`` object as JSON-LD for SEO, which is a
stable, machine-readable contract — far more robust than CSS-selector scraping.
This module parses those blocks (plus a couple of ``og:`` meta fallbacks) into
the fields a :class:`~sneakerrag.models.Listing` needs.
"""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any, Iterator

_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
_META_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\']([^"\']+)["\'][^>]+content=["\']([^"\']*)["\']', re.I
)

IN_STOCK_TOKENS = ("instock", "in_stock", "limitedavailability", "onlineonly", "preorder")


def iter_jsonld(html: str) -> Iterator[Any]:
    for match in _SCRIPT_RE.finditer(html or ""):
        blob = unescape(match.group(1)).strip()
        blob = re.sub(r"^\s*<!--|-->\s*$", "", blob)
        try:
            yield json.loads(blob)
        except json.JSONDecodeError:
            # Some sites concatenate objects; try the outermost array/object.
            for candidate in re.findall(r"\{.*\}|\[.*\]", blob, re.S):
                try:
                    yield json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue


def _walk(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _types(node: dict) -> set[str]:
    raw = node.get("@type") or node.get("type") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(t).lower() for t in raw}


def find_products(html: str) -> list[dict]:
    products = []
    for blob in iter_jsonld(html):
        for node in _walk(blob):
            if "product" in _types(node) and (node.get("name") or node.get("sku")):
                products.append(node)
    return products


def _first_offer(node: dict) -> dict:
    offers = node.get("offers")
    candidates: list[dict] = []
    for item in _walk(offers) if offers is not None else []:
        if isinstance(item, dict) and ("price" in item or "lowPrice" in item):
            candidates.append(item)
    if not candidates:
        return {}
    # Cheapest offer on the page is the one a shopper would actually take.
    def price_of(o: dict) -> float:
        try:
            return float(str(o.get("price") or o.get("lowPrice") or 1e9).replace(",", ""))
        except ValueError:
            return 1e9
    return min(candidates, key=price_of)


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("@id") or "")
    if isinstance(value, list):
        return _text(value[0]) if value else ""
    return "" if value is None else str(value)


def parse_product(node: dict) -> dict:
    """Flatten a schema.org Product node into listing-shaped fields."""
    offer = _first_offer(node)
    availability = _text(offer.get("availability")).lower().replace("http://schema.org/", "")
    sizes: list[str] = []
    for item in _walk(node):
        if isinstance(item, dict) and str(item.get("name", "")).lower() in ("size", "shoe size"):
            value = item.get("value")
            if isinstance(value, list):
                sizes.extend(str(v) for v in value)
            elif value:
                sizes.append(str(value))
    return {
        "title": _text(node.get("name")),
        "brand": _text(node.get("brand")),
        "style_code": _text(node.get("mpn") or node.get("productID") or ""),
        "retailer_sku": _text(node.get("sku")),
        "colorway": _text(node.get("color")),
        "gender": _text(node.get("audience") or node.get("gender")),
        "image": _text(node.get("image")),
        "url": _text(offer.get("url") or node.get("url")),
        "price": offer.get("price") or offer.get("lowPrice"),
        "list_price": (offer.get("priceSpecification") or {}).get("price")
        if isinstance(offer.get("priceSpecification"), dict) else None,
        "currency": _text(offer.get("priceCurrency")) or "USD",
        "in_stock": any(token in availability for token in IN_STOCK_TOKENS) if availability else True,
        "sizes": sizes,
        "description": _text(node.get("description"))[:400],
    }


def meta_fallback(html: str) -> dict:
    """og:/product: meta tags, for pages without JSON-LD."""
    meta = {k.lower(): v for k, v in _META_RE.findall(html or "")}
    out: dict[str, Any] = {}
    if "og:title" in meta:
        out["title"] = unescape(meta["og:title"])
    for key in ("product:price:amount", "og:price:amount", "twitter:data1"):
        if meta.get(key):
            out["price"] = meta[key]
            break
    for key in ("product:price:currency", "og:price:currency"):
        if meta.get(key):
            out["currency"] = meta[key]
            break
    if "og:image" in meta:
        out["image"] = meta["og:image"]
    if "og:url" in meta:
        out["url"] = meta["og:url"]
    return out


def extract_listing_fields(html: str) -> dict:
    """Best-effort merge of JSON-LD and meta-tag data for one product page."""
    products = find_products(html)
    data = parse_product(products[0]) if products else {}
    for key, value in meta_fallback(html).items():
        if not data.get(key):
            data[key] = value
    return data

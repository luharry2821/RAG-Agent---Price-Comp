"""Core data types for the sneaker price-comparison RAG agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Listing:
    """One product page on one retailer.

    A listing is the atomic unit we ingest, embed and cite. Several listings
    across different retailers may describe the same physical shoe; grouping
    them is the job of :mod:`sneakerrag.matching`.
    """

    source: str                      # site key, e.g. "goat"
    source_name: str                 # display name, e.g. "GOAT"
    url: str
    title: str
    source_kind: str = "retail"      # brand | retail | resale
    brand: str = ""                  # normalized: nike | adidas | new balance
    model: str = ""                  # e.g. "air max 90"
    colorway: str = ""               # e.g. "white/black"
    style_code: str = ""             # manufacturer SKU, e.g. "HM0089-100"
    retailer_sku: str = ""           # retailer's own id
    price: float = 0.0               # headline price; on resale sites the lowest ask
    list_price: float | None = None  # MSRP / "was" price
    currency: str = "USD"
    shipping: float | None = None    # None = unknown, 0.0 = free
    fees: float | None = None        # marketplace processing / authentication fee
    in_stock: bool = True
    sizes: list[str] = field(default_factory=list)
    size_prices: dict[str, float] = field(default_factory=dict)
    gender: str = ""                 # men | women | unisex | kids
    condition: str = "new"           # new | used
    image: str = ""
    scraped_at: str = field(default_factory=utcnow)
    raw: dict[str, Any] = field(default_factory=dict)
    listing_id: str = ""

    def __post_init__(self) -> None:
        if not self.listing_id:
            self.listing_id = self.compute_id()

    def compute_id(self) -> str:
        seed = f"{self.source}|{self.url}".lower()
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    # -- pricing helpers -------------------------------------------------
    #
    # Retail sites charge one price for every size. Resale marketplaces price
    # each size separately — a 10.5 and a 13 of the same SKU are different
    # numbers, often by a lot — so size-aware pricing is the norm here, not an
    # edge case, and ``price`` is only the "from" figure.

    def price_for(self, size: str = "") -> float:
        """Price for one size, falling back to the headline price."""
        if size and self.size_prices:
            key = size.strip().lower().lstrip("0")
            for candidate, value in self.size_prices.items():
                if candidate.strip().lower().lstrip("0") == key:
                    return value
        return self.price

    @property
    def extras(self) -> float:
        """Shipping plus marketplace fees, in dollars."""
        return round((self.shipping or 0.0) + (self.fees or 0.0), 2)

    def total_for(self, size: str = "") -> float:
        """What the buyer actually pays for that size, delivered."""
        return round(self.price_for(size) + self.extras, 2)

    @property
    def total_price(self) -> float:
        """Delivered price at the headline (lowest) ask."""
        return self.total_for()

    @property
    def is_resale(self) -> bool:
        return self.source_kind == "resale"

    def discount_pct_for(self, size: str = "") -> float:
        """Percent below MSRP for one size (resale asks often sit above it)."""
        price = self.price_for(size)
        if not self.list_price or self.list_price <= 0 or price >= self.list_price:
            return 0.0
        return round((1 - price / self.list_price) * 100, 1)

    @property
    def discount_pct(self) -> float:
        return self.discount_pct_for()

    def premium_pct_for(self, size: str = "") -> float:
        """Percent *above* MSRP — the resale markup on a hyped size."""
        price = self.price_for(size)
        if not self.list_price or self.list_price <= 0 or price <= self.list_price:
            return 0.0
        return round((price / self.list_price - 1) * 100, 1)

    def has_size(self, size: str) -> bool:
        if not size:
            return True
        want = size.strip().lower().lstrip("0")
        pool = list(self.sizes) + list(self.size_prices)
        return any(s.strip().lower().lstrip("0") == want for s in pool)

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Listing":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def summary(self) -> str:
        """Compact one-line form used as the document text for retrieval."""
        bits = [self.brand, self.model, self.colorway, self.style_code,
                self.gender, self.source_name, self.title]
        return " ".join(b for b in bits if b)


@dataclass
class Product:
    """A cluster of listings believed to be the same shoe/SKU."""

    product_key: str
    brand: str
    model: str
    colorway: str = ""
    style_code: str = ""
    gender: str = ""
    listings: list[Listing] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        name = f"{self.brand.title()} {self.model.title()}".strip()
        if self.colorway:
            name += f" — {self.colorway.title()}"
        return name

    def offers(self, *, size: str = "", in_stock_only: bool = True,
               include_shipping: bool = True, condition: str = "") -> list[Listing]:
        """Listings that satisfy the buyer's constraints, cheapest first.

        Ranking is by the price of the *requested size* where the seller prices
        per size, so a marketplace with a cheap size 8 does not win a size 12
        question.
        """
        pool = [l for l in self.listings
                if (not in_stock_only or l.in_stock)
                and l.has_size(size)
                and (not condition or l.condition == condition)]
        if include_shipping:
            key = lambda l: (l.total_for(size), l.price_for(size))       # noqa: E731
        else:
            key = lambda l: (l.price_for(size), l.total_for(size))       # noqa: E731
        return sorted(pool, key=key)

    def best_offer(self, **kw: Any) -> Listing | None:
        offers = self.offers(**kw)
        return offers[0] if offers else None

    def price_range(self, size: str = "") -> tuple[float, float] | None:
        prices = [l.total_for(size) for l in self.listings if l.in_stock and l.has_size(size)]
        return (min(prices), max(prices)) if prices else None

    def savings(self, **kw: Any) -> float:
        """Money saved by buying the cheapest listing vs the priciest."""
        offers = self.offers(**kw)
        if len(offers) < 2:
            return 0.0
        size = kw.get("size", "")
        return round(offers[-1].total_for(size) - offers[0].total_for(size), 2)

    @property
    def has_resale_listings(self) -> bool:
        return any(l.is_resale for l in self.listings)

    @property
    def at_retail(self) -> bool:
        """True when a brand/retail store still stocks it (vs resale only)."""
        return any(not l.is_resale and l.in_stock for l in self.listings)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class QuerySpec:
    """Structured form of a natural-language shopping question."""

    text: str = ""
    brands: list[str] = field(default_factory=list)
    terms: str = ""              # model keywords, e.g. "air max 90"
    style_code: str = ""
    size: str = ""
    gender: str = ""
    max_price: float | None = None
    min_discount: float | None = None
    condition: str = ""          # "" = any, else new | used
    in_stock_only: bool = True
    include_shipping: bool = True
    top_k: int = 40
    intent: str = "compare"      # compare | deals | lookup

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        bits = []
        if self.brands:
            bits.append("brand=" + "/".join(self.brands))
        if self.terms:
            bits.append(f"model~{self.terms}")
        if self.style_code:
            bits.append(f"sku={self.style_code}")
        if self.size:
            bits.append(f"size={self.size}")
        if self.gender:
            bits.append(f"gender={self.gender}")
        if self.max_price:
            bits.append(f"under ${self.max_price:g}")
        if self.condition:
            bits.append(f"condition={self.condition}")
        return ", ".join(bits) or "(no filters)"


@dataclass
class Citation:
    n: int
    source_name: str
    url: str
    price: float
    currency: str = "USD"
    note: str = ""            # why this listing can't be bought as asked

    def render(self) -> str:
        note = f" ({self.note})" if self.note else ""
        return (f"[{self.n}] {self.source_name} — {fmt_money(self.price, self.currency)}"
                f"{note} — {self.url}")


@dataclass
class Answer:
    question: str
    text: str
    products: list[Product] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    spec: QuerySpec | None = None
    generator: str = "template"   # "claude" | "template"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "generator": self.generator,
            "spec": self.spec.to_dict() if self.spec else None,
            "citations": [asdict(c) for c in self.citations],
            "products": [p.to_dict() for p in self.products],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


CURRENCY_SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€", "CAD": "CA$"}


def fmt_money(amount: float | None, currency: str = "USD") -> str:
    if amount is None:
        return "n/a"
    sym = CURRENCY_SYMBOLS.get(currency.upper(), currency.upper() + " ")
    return f"{sym}{amount:,.2f}"

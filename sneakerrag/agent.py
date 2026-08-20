"""The RAG agent: question -> retrieve -> group by SKU -> cheapest -> answer.

Pipeline
--------
1. **Understand** the question (heuristics always; Claude refines when available).
2. **Retrieve** candidate listings from the hybrid index, ignoring price/size
   filters at this stage so the comparison table stays complete.
3. **Group** candidates into products with :mod:`sneakerrag.matching`.
4. **Compare** prices inside each product, applying the buyer's constraints.
5. **Answer** with Claude over the retrieved evidence, or a deterministic
   template when Claude is not configured. Every price claim is cited.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .embeddings import get_embedder
from .llm import LLMUnavailable, get_llm, parse_json_response
from .matching import cluster_listings
from .models import Answer, Citation, Listing, Product, QuerySpec, fmt_money
from .aliases import term_sources
from .normalize import (
    BRAND_ALIASES,
    extract_style_code,
    normalize_brand,
    normalize_style_code,
    squash,
)
from .retrieve import GENERIC_TERMS, VectorIndex
from .sources import get_sources
from .store import Catalog, DEFAULT_DB

# --------------------------------------------------------------------------
# Query understanding
# --------------------------------------------------------------------------

# Tried in order: "size 10.5" beats "…90 size…", so a model number in the
# query ("Air Max 90 size 10.5") can't be mistaken for the shoe size.
_SIZE_PATTERNS = (
    re.compile(r"\bsize\s*(?:us\s*)?(\d{1,2}(?:\.5)?)\b", re.I),
    re.compile(r"\b(\d{1,2}(?:\.5)?)\s*(?:us|uk)\s*size\b", re.I),
    re.compile(r"\bin\s+an?\s+(\d{1,2}(?:\.5)?)\b", re.I),
)
_UNDER_RE = re.compile(
    r"\b(?:under|below|less than|cheaper than|up to|no more than|"
    r"max(?:imum)?\s+(?:of|price|budget)|budget\s+of|within)\s*\$\s*?(\d[\d,]*(?:\.\d{1,2})?)"
    r"|\b(?:under|below|less than|cheaper than|up to|no more than|"
    r"max(?:imum)?\s+(?:of|price|budget)|budget\s+of|within)\s+(\d[\d,]*(?:\.\d{1,2})?)\s*(?:dollars|usd|bucks)\b",
    re.I,
)
_DISCOUNT_RE = re.compile(r"\b(\d{1,2})\s*%\s*(?:off|discount)", re.I)
_STOPWORDS = {
    "what", "whats", "which", "where", "who", "the", "a", "an", "is", "are",
    "can", "i", "me", "my", "buy", "get", "find", "cheapest", "cheap", "best",
    "price", "prices", "deal", "deals", "lowest", "site", "sites", "website",
    "websites", "for", "of", "on", "in", "at", "to", "and", "or", "with",
    "show", "give", "compare", "comparison", "shoe", "shoes", "sneaker",
    "sneakers", "pair", "please", "cost", "costs", "much", "how", "right",
    "now", "currently", "available", "stock", "size", "sizes", "any", "all",
    "list", "listing", "listings", "that", "has", "have", "including",
    "include", "shipping", "delivery", "delivered", "free", "under", "below",
    "less", "than", "cheaper", "up", "no", "more", "budget", "within", "us",
}

_INTENT_HINTS = (
    ("deals", re.compile(r"\b(deal|deals|sale|sales|discount|clearance|drop|drops|off)\b", re.I)),
    ("lookup", re.compile(r"\b(style code|sku|[A-Z]{2}\d{4}-\d{3}|[A-Z]\d{5})\b")),
)

_PARSE_SYSTEM = """You convert sneaker shopping questions into a JSON filter object.

Return ONLY a JSON object with these keys (omit a key when the question does not mention it):
  brands: array, any of ["nike", "adidas", "new balance"]
  terms: string - the model/silhouette keywords only, e.g. "air max 90", "990v6 grey"
  style_code: string - manufacturer style code if one appears, e.g. "DD1391-100"
  size: string - US size as a number string, e.g. "10.5"
  gender: string - one of "men", "women", "kids", "unisex"
  max_price: number - budget ceiling in USD
  min_discount: number - required percent off
  condition: string - "new" or "used" if the shopper asked for one
  in_stock_only: boolean
  include_shipping: boolean - true if the user cares about delivered cost
  intent: string - "compare" (default), "deals", or "lookup"

No prose, no markdown fences."""


def heuristic_spec(question: str) -> QuerySpec:
    """Regex/keyword parse — always runs, and is the fallback when Claude is off."""
    text = question or ""
    spec = QuerySpec(text=text)

    for brand, aliases in BRAND_ALIASES.items():
        if any(re.search(rf"\b{re.escape(a)}\b", text, re.I) for a in aliases):
            spec.brands.append(brand)

    code = extract_style_code(text)
    if code:
        spec.style_code = normalize_style_code(code)

    for pattern in _SIZE_PATTERNS:
        m = pattern.search(text)
        if m:
            spec.size = m.group(1).lstrip("0")
            break

    m = _UNDER_RE.search(text)
    if m:
        spec.max_price = float((m.group(1) or m.group(2)).replace(",", ""))

    m = _DISCOUNT_RE.search(text)
    if m:
        spec.min_discount = float(m.group(1))

    if re.search(r"\bwomen'?s?\b|\bwmns\b", text, re.I):
        spec.gender = "women"
    elif re.search(r"\bmen'?s?\b", text, re.I):
        spec.gender = "men"
    elif re.search(r"\bkids?\b|\bjunior\b|\bgrade school\b", text, re.I):
        spec.gender = "kids"

    if re.search(r"\b(include|with|including)\s+(shipping|delivery)\b", text, re.I):
        spec.include_shipping = True
    if re.search(r"\b(ignore|excluding|without|before)\s+(shipping|delivery)\b", text, re.I):
        spec.include_shipping = False
    if re.search(r"\b(sold out|out of stock|any listing)\b", text, re.I):
        spec.in_stock_only = False

    if re.search(r"\b(brand new|deadstock|ds|unworn|new only|new pair)\b", text, re.I):
        spec.condition = "new"
    elif re.search(r"\b(used|pre-?owned|worn|beaters)\b", text, re.I):
        spec.condition = "used"

    for intent, pattern in _INTENT_HINTS:
        if pattern.search(text):
            spec.intent = intent
            break

    spec.terms = _model_terms(text, spec)
    return spec


def _model_terms(text: str, spec: QuerySpec) -> str:
    """Strip brand names, prices and question words down to model keywords."""
    t = squash(text)
    for aliases in BRAND_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            t = re.sub(rf"\b{re.escape(alias)}\b", " ", t)
    t = _UNDER_RE.sub(" ", t)
    t = re.sub(r"\$\s*\d[\d,.]*", " ", t)
    if spec.max_price is not None:
        # squash() strips the "$", so the budget survives as a bare number.
        for form in {f"{spec.max_price:g}", f"{spec.max_price:.2f}"}:
            t = re.sub(rf"\b{re.escape(form)}\b", " ", t)
    if spec.size:
        t = re.sub(rf"\bsize\s*{re.escape(spec.size)}\b|\b{re.escape(spec.size)}\b", " ", t)
    tokens = [w for w in re.split(r"[^a-z0-9.]+", t) if w and w not in _STOPWORDS]
    return " ".join(tokens).strip()


def refine_spec_with_llm(question: str, spec: QuerySpec, llm: Any) -> QuerySpec:
    """Let Claude fill in what the regexes missed; heuristics win on conflicts
    for anything they matched confidently (codes, explicit budgets)."""
    if not getattr(llm, "available", False):
        return spec
    try:
        raw = llm.complete(_PARSE_SYSTEM, f"Question: {question}", max_tokens=600)
    except LLMUnavailable:
        return spec
    data = parse_json_response(raw)
    if not data:
        return spec

    out = replace(spec)
    brands = [normalize_brand(b) for b in data.get("brands", []) if normalize_brand(b)]
    if brands:
        out.brands = sorted(set(out.brands) | set(brands))
    if not out.style_code and data.get("style_code"):
        out.style_code = normalize_style_code(str(data["style_code"]))
    if not out.size and data.get("size"):
        out.size = str(data["size"]).strip()
    if not out.gender and data.get("gender") in ("men", "women", "kids", "unisex"):
        out.gender = data["gender"]
    if out.max_price is None and isinstance(data.get("max_price"), (int, float)):
        out.max_price = float(data["max_price"])
    if out.min_discount is None and isinstance(data.get("min_discount"), (int, float)):
        out.min_discount = float(data["min_discount"])
    if not out.condition and data.get("condition") in ("new", "used"):
        out.condition = data["condition"]
    if isinstance(data.get("in_stock_only"), bool):
        out.in_stock_only = data["in_stock_only"]
    if isinstance(data.get("include_shipping"), bool):
        out.include_shipping = data["include_shipping"]
    if data.get("intent") in ("compare", "deals", "lookup"):
        out.intent = data["intent"]
    llm_terms = str(data.get("terms") or "").strip()
    if llm_terms and len(llm_terms) >= len(out.terms) / 2:
        out.terms = squash(llm_terms)
    return out


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

_ANSWER_SYSTEM = """You are a sneaker price-comparison assistant.

You are given a shopper's question and a table of retailer listings that were
retrieved from a catalogue. Rules:

- Use ONLY the listings provided. Never invent a retailer, price, size or URL.
- Lead with the single cheapest qualifying listing: seller, delivered price, and why it
  wins (sale price, free shipping, no fees, cheaper size).
- Then compare the other sellers briefly so the saving is visible.
- Cite every price with the bracketed number shown next to the listing, e.g. [2].
- Prices on resale marketplaces are per size and exclude fees until checkout; always
  quote the delivered price for the size asked about, and say when a listing is used
  rather than new.
- If the shoe is sold out at the brand store and only resale listings remain, say so,
  and say how far above MSRP the asks are.
- Call out constraints that could not be met (size unavailable, out of stock,
  nothing under the stated budget) instead of silently ignoring them.
- Be concise: a short paragraph plus a compact list. No preamble, no markdown headings."""


class SneakerAgent:
    """Retrieval-augmented price comparison over the listing catalogue."""

    def __init__(self, catalog: Catalog | None = None, index: VectorIndex | None = None,
                 llm: Any | None = None, db_path: Path | str = DEFAULT_DB,
                 use_llm: bool = True) -> None:
        self.catalog = catalog if catalog is not None else Catalog(db_path)
        self.llm = llm if llm is not None else get_llm(enabled=use_llm)
        self.index = index if index is not None else self._build_index()

    # -- index ------------------------------------------------------------
    def _build_index(self) -> VectorIndex:
        embedder = get_embedder()
        index = VectorIndex(embedder=embedder)
        listings, vectors = self.catalog.listings_with_vectors()
        if not listings:
            return index
        usable = [v if v and len(v) == embedder.dim else None for v in vectors]
        if all(v is not None for v in usable):
            index.add(listings, usable)          # reuse cached embeddings
        else:
            index.add(listings)
        return index

    def reload(self) -> None:
        self.index = self._build_index()

    # -- ingest -----------------------------------------------------------
    def ingest(self, sources: Iterable[Any] | None = None, queries: Sequence[str] | None = None,
               limit: int = 50) -> dict[str, Any]:
        """Pull listings from sources into the catalogue and re-embed."""
        adapters = list(sources) if sources is not None else get_sources()
        collected: list[Listing] = []
        per_source: dict[str, int] = {}
        for adapter in adapters:
            found = (adapter.collect(queries, limit=limit) if queries
                     else adapter.search("", limit=limit))
            per_source[adapter.key] = len(found)
            collected.extend(found)

        embedder = get_embedder()
        vectors = embedder.embed_many([l.summary() for l in collected])
        stats = self.catalog.upsert_listings(collected, vectors)
        self.reload()
        return {"per_source": per_source, "total": len(collected), **stats}

    # -- retrieval + comparison -------------------------------------------
    def understand(self, question: str) -> QuerySpec:
        return refine_spec_with_llm(question, heuristic_spec(question), self.llm)

    def _retrieval_spec(self, spec: QuerySpec) -> QuerySpec:
        """Candidate generation deliberately drops buyer constraints.

        A "cheapest under $120 in size 10" question still needs the pricier
        listings retrieved, otherwise the comparison table has nothing to
        compare against and we cannot tell the shopper what they are saving.
        """
        relaxed = replace(spec)
        relaxed.max_price = None
        relaxed.min_discount = None
        relaxed.size = ""
        relaxed.in_stock_only = False
        return relaxed

    def search_products(self, spec: QuerySpec) -> list[Product]:
        """Retrieve listings and group them into ranked products."""
        # Brand is applied as a hard filter, so repeating it in the free-text
        # query would only dilute the model terms and inflate term coverage.
        query = " ".join(x for x in (spec.terms, spec.style_code) if x) or spec.text
        hits = self.index.search(query or spec.text, self._retrieval_spec(spec), top_k=spec.top_k)
        if not hits:
            return []
        rank = {h.listing.listing_id: h.score for h in hits}
        products = cluster_listings([h.listing for h in hits])

        # A retrieved listing may have siblings that missed the top-k cut; pull
        # the full cluster in so the price table is complete.
        known = {l.listing_id for p in products for l in p.listings}
        extras: list[Listing] = []
        codes = {p.style_code for p in products if p.style_code}
        for listing in self.catalog.listings():
            if listing.listing_id in known:
                continue
            if listing.style_code and normalize_style_code(listing.style_code) in codes:
                extras.append(listing)
        if extras:
            products = cluster_listings([l for p in products for l in p.listings] + extras)

        products.sort(key=lambda p: max(rank.get(l.listing_id, 0.0) for l in p.listings), reverse=True)
        return products

    def compare(self, question: str, spec: QuerySpec | None = None,
                limit: int = 3) -> tuple[QuerySpec, list[Product]]:
        spec = spec or self.understand(question)
        products = self.search_products(spec)
        if spec.intent == "deals":
            products.sort(key=lambda p: -max((l.discount_pct for l in p.listings), default=0.0))
        return spec, products[:limit]

    # -- answering ---------------------------------------------------------
    def build_context(self, products: Sequence[Product], spec: QuerySpec
                      ) -> tuple[str, list[Citation]]:
        """Render retrieved evidence as a numbered, citable block."""
        lines: list[str] = []
        citations: list[Citation] = []
        n = 0
        for product in products:
            header = product.display_name
            if product.style_code:
                header += f"  (style {product.style_code})"
            if product.style_code or product.listings:
                header += f"  — MSRP {fmt_money(_msrp(product), 'USD')}" if _msrp(product) else ""
            if not product.at_retail and product.has_resale_listings:
                header += "  [sold out at the brand store; resale only]"
            lines.append(header)
            gap = match_gap(spec, product)
            if gap:
                lines.append(f"  NOTE: {gap}")
            # Everything in the cluster is cited, including listings that fail
            # the buyer's constraints — the flags below explain each exclusion.
            # Cite every listing in the cluster (size filter off), but order
            # them by what the requested size actually costs.
            everything = sorted(product.offers(size="", in_stock_only=False,
                                               include_shipping=spec.include_shipping),
                                key=lambda l: l.total_for(spec.size))
            for listing in everything:
                n += 1
                notes = []
                if not listing.in_stock:
                    notes.append("out of stock")
                if spec.size and not listing.has_size(spec.size):
                    # The quoted price is this seller's lowest ask in some other
                    # size — say so, or the citation list reads as a better deal.
                    notes.append(f"no US {spec.size}; price shown is their lowest ask")
                if listing.condition != "new":
                    notes.append(listing.condition)
                citations.append(Citation(n=n, source_name=listing.source_name, url=listing.url,
                                          price=listing.total_for(spec.size),
                                          currency=listing.currency, note=", ".join(notes)))
                lines.append(f"  [{n}] {describe_listing(listing, spec)}")
                lines.append(f"      sizes: {', '.join(listing.sizes) or 'n/a'}")
                lines.append(f"      {listing.url}")
            lines.append("")
        return "\n".join(lines).strip(), citations

    def answer(self, question: str, limit: int = 3, spec: QuerySpec | None = None) -> Answer:
        spec, products = self.compare(question, spec=spec, limit=limit)
        if not products:
            return Answer(question=question, spec=spec, generator="template",
                          text=("No listings matched that. The catalogue currently covers "
                                f"{self.catalog.stats()['listings']} listings across "
                                "Nike, adidas and New Balance — try a broader model name, "
                                "or run `ingest` to refresh it."))

        context, citations = self.build_context(products, spec)
        text = ""
        generator = "template"
        if getattr(self.llm, "available", False):
            user = (f"Shopper question: {question}\n"
                    f"Parsed constraints: {spec.describe()}\n\n"
                    f"Retrieved listings:\n{context}")
            try:
                text = self.llm.complete(_ANSWER_SYSTEM, user, max_tokens=1500)
                generator = "claude"
            except LLMUnavailable:
                text = ""
        if not text:
            text = render_template_answer(question, spec, products, citations)
            generator = "template"
        return Answer(question=question, text=text, products=list(products),
                      citations=citations, spec=spec, generator=generator)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

_MODEL_NUMBER = re.compile(r"\b\d{2,4}(?:v\d)?\b")


def match_gap(spec: QuerySpec, product: Product) -> str:
    """Name the parts of the question this product does not actually answer.

    Two shoes can share a silhouette and differ in the only thing the shopper
    named: an Air Jordan 1 Retro High OG "Love Letter" and the same shoe in
    "Chicago Lost and Found" agree on every word except the two that matter.
    Quoting one for the other without saying so answers a question nobody
    asked — so any typed word the product cannot account for is reported.
    """
    query = (spec.terms or spec.text or "").strip()
    if not query:
        return ""

    haystack = " ".join([product.model, product.colorway, product.style_code]
                        + [l.title for l in product.listings]).lower()
    known = set(re.split(r"[^a-z0-9]+", haystack))
    sources = term_sources(query)

    # A typed word counts as answered if it, or anything it expands to
    # ("panda" -> white/black), appears in the product.
    satisfied: dict[str, bool] = {}
    for term, origins in sources.items():
        for origin in origins:
            satisfied[origin] = satisfied.get(origin, False) or term in known

    missing = [word for word, found in satisfied.items()
               if not found and word not in GENERIC_TERMS and len(word) > 1]
    if not missing:
        return ""

    numbers = [word for word in missing if _MODEL_NUMBER.fullmatch(word)]
    detail = " ".join(sorted(missing, key=lambda w: query.find(w)))
    if numbers and not any(_MODEL_NUMBER.findall(product.model)):
        return ""                       # nothing numeric to contradict
    return (f"No exact match for \"{detail}\" in the catalogue — "
            f"showing the closest, {product.display_name}.")


def _msrp(product: Product) -> float | None:
    """Best available MSRP for a product: the brand store price, else a
    retailer's "was" price."""
    for listing in product.listings:
        if listing.source_kind == "brand" and not listing.discount_pct:
            return listing.price
    quoted = [l.list_price for l in product.listings if l.list_price]
    return max(quoted) if quoted else None


def price_note(listing: Listing, spec: QuerySpec) -> str:
    """'$165.00 for a US 10.5' vs '$130.00' — size-aware price phrasing."""
    price = listing.price_for(spec.size)
    text = fmt_money(price, listing.currency)
    if spec.size and listing.size_prices:
        text += f" for a US {spec.size}"
    elif listing.size_prices:
        text += " lowest ask"
    return text


def describe_listing(listing: Listing, spec: QuerySpec) -> str:
    """One line covering price, extras, condition and any failed constraint."""
    parts = []
    if listing.shipping == 0:
        parts.append("free shipping")
    elif listing.shipping:
        parts.append(f"+{fmt_money(listing.shipping, listing.currency)} shipping")
    else:
        parts.append("shipping unknown")
    if listing.fees:
        parts.append(f"+{fmt_money(listing.fees, listing.currency)} fees")

    flags = []
    if not listing.in_stock:
        flags.append("OUT OF STOCK")
    if spec.size and not listing.has_size(spec.size):
        flags.append(f"size {spec.size} unavailable")
    if listing.condition != "new":
        flags.append(listing.condition)
    discount = listing.discount_pct_for(spec.size)
    premium = listing.premium_pct_for(spec.size)
    if discount:
        flags.append(f"{discount:g}% off {fmt_money(listing.list_price, listing.currency)}")
    elif premium:
        flags.append(f"{premium:g}% above the {fmt_money(listing.list_price, listing.currency)} MSRP")

    return (f"{listing.source_name}: {price_note(listing, spec)}"
            f" ({'; '.join(parts)}; delivered "
            f"{fmt_money(listing.total_for(spec.size), listing.currency)})"
            + (f" — {', '.join(flags)}" if flags else ""))


# --------------------------------------------------------------------------
# Deterministic answer writer (used whenever Claude is unavailable)
# --------------------------------------------------------------------------

def render_template_answer(question: str, spec: QuerySpec, products: Sequence[Product],
                           citations: Sequence[Citation]) -> str:
    """The offline answer writer: same evidence, no model in the loop."""
    by_url = {c.url: c for c in citations}
    where = f" in a US {spec.size}" if spec.size else ""
    out: list[str] = []

    def cite(listing: Listing) -> str:
        found = by_url.get(listing.url)
        return f" [{found.n}]" if found else ""

    for product in products:
        title = product.display_name
        if product.style_code:
            title += f" (style {product.style_code})"
        msrp = _msrp(product)
        if msrp:
            title += f" — MSRP {fmt_money(msrp)}"
        out.append(title)

        gap = match_gap(spec, product)
        if gap:
            out.append(f"  {gap}")
        if not product.at_retail and product.has_resale_listings:
            out.append("  Sold out at the brand store — the listings below are resale asks.")

        offers = product.offers(size=spec.size, in_stock_only=spec.in_stock_only,
                                condition=spec.condition,
                                include_shipping=spec.include_shipping)
        if not offers:
            wanted = [x for x in (f"size {spec.size}" if spec.size else "",
                                  "in stock" if spec.in_stock_only else "",
                                  spec.condition) if x]
            out.append(f"  No listing matches {' and '.join(wanted) or 'those constraints'} "
                       f"across the {len({l.source for l in product.listings})} sites checked.")
            out.append("")
            continue

        if spec.max_price is not None:
            affordable = [o for o in offers
                          if (o.total_for(spec.size) if spec.include_shipping
                              else o.price_for(spec.size)) <= spec.max_price]
        else:
            affordable = offers

        if not affordable:
            cheapest = offers[0]
            out.append(f"  Nothing{where} under {fmt_money(spec.max_price)}. Cheapest is "
                       f"{describe_listing(cheapest, spec)}{cite(cheapest)}")
            out.append(f"  {cheapest.url}")
            out.append("")
            continue

        best = affordable[0]
        out.append(f"  Cheapest{where}: {describe_listing(best, spec)}{cite(best)}")
        out.append(f"  {best.url}")

        saving = round(offers[-1].total_for(spec.size) - best.total_for(spec.size), 2)
        if saving > 0:
            out.append(f"  Saves {fmt_money(saving)} vs the priciest of the "
                       f"{len(offers)} listings compared.")

        for other in offers[1:]:
            out.append(f"    - {describe_listing(other, spec)}{cite(other)}")

        # Explain the gaps in the citation numbering rather than leaving them.
        qualifying = {l.listing_id for l in offers}
        for skipped in product.listings:
            if skipped.listing_id in qualifying:
                continue
            reasons = []
            if not skipped.in_stock:
                reasons.append("out of stock")
            if spec.size and not skipped.has_size(spec.size):
                reasons.append(f"no size {spec.size}")
            if spec.condition and skipped.condition != spec.condition:
                reasons.append(f"{skipped.condition}, not {spec.condition}")
            out.append(f"    - excluded {skipped.source_name} "
                       f"({', '.join(reasons) or 'filtered out'}){cite(skipped)}")
        out.append("")

    footers = []
    if spec.size:
        footers.append(f"Size filter: US {spec.size}")
    if spec.condition:
        footers.append(f"condition: {spec.condition}")
    if spec.max_price is not None:
        footers.append(f"budget: {fmt_money(spec.max_price)}"
                       f"{' delivered' if spec.include_shipping else ' before shipping and fees'}")
    if footers:
        out.append(". ".join(footers) + ".")
    return "\n".join(out).strip()

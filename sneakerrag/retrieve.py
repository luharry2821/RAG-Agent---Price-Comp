"""Hybrid retrieval: BM25 lexical scoring blended with vector similarity.

Pure vector search is weak on this data — "990v6" and "990v5" are near
identical in embedding space — while pure keyword search misses paraphrases
("running shoes for wide feet"). The index runs both and blends the normalized
scores, then applies hard structured filters (brand, size, price, stock) that a
shopping query must respect exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .embeddings import Embedder, cosine, get_embedder
from .models import Listing, QuerySpec
from .normalize import normalize_style_code, squash

K1 = 1.5
B = 0.75
DEFAULT_ALPHA = 0.6      # weight of the lexical (BM25) half


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9.]+", squash(text)) if t]


@dataclass
class Hit:
    listing: Listing
    score: float
    lexical: float = 0.0
    vector: float = 0.0
    boost: float = 0.0


class VectorIndex:
    """In-memory hybrid index over listings.

    Small by design: a few thousand listings is the realistic size of a
    price-comparison catalogue for three brands, and keeping it in memory means
    no external vector database to run.
    """

    def __init__(self, embedder: Embedder | None = None, alpha: float = DEFAULT_ALPHA) -> None:
        self.embedder = embedder or get_embedder()
        self.alpha = alpha
        self.listings: list[Listing] = []
        self.vectors: list[list[float]] = []
        self.docs: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.avg_len: float = 0.0

    # -- build ------------------------------------------------------------
    def add(self, listings: Iterable[Listing], vectors: Sequence[Sequence[float]] | None = None) -> None:
        items = list(listings)
        if not items:
            return
        vecs = list(vectors) if vectors else self.embedder.embed_many([l.summary() for l in items])
        for listing, vec in zip(items, vecs):
            self.listings.append(listing)
            self.vectors.append(list(vec))
            tokens = tokenize(listing.summary())
            self.docs.append(tokens)
        self._reindex()

    def _reindex(self) -> None:
        self.df = {}
        total = 0
        for tokens in self.docs:
            total += len(tokens)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = (total / len(self.docs)) if self.docs else 0.0

    def __len__(self) -> int:
        return len(self.listings)

    # -- scoring ----------------------------------------------------------
    def _bm25(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        tokens = self.docs[doc_idx]
        if not tokens:
            return 0.0
        n = len(self.docs)
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        dl = len(tokens)
        for qt in query_tokens:
            f = freq.get(qt, 0)
            if not f:
                continue
            idf = math.log(1 + (n - self.df.get(qt, 0) + 0.5) / (self.df.get(qt, 0) + 0.5))
            denom = f + K1 * (1 - B + B * dl / (self.avg_len or 1))
            score += idf * (f * (K1 + 1)) / denom
        return score

    def search(self, query: str, spec: QuerySpec | None = None, top_k: int = 40) -> list[Hit]:
        spec = spec or QuerySpec(text=query)
        q_tokens = tokenize(query)
        q_vec = self.embedder.embed(query)
        code = normalize_style_code(spec.style_code)

        raw: list[tuple[int, float, float, float]] = []
        for i, listing in enumerate(self.listings):
            if not _passes_filters(listing, spec):
                continue
            lex = self._bm25(q_tokens, i)
            vec = cosine(q_vec, self.vectors[i])
            boost = 0.0
            if code and normalize_style_code(listing.style_code) == code:
                boost += 5.0                     # exact SKU beats everything
            if spec.brands and listing.brand in spec.brands:
                boost += 0.15
            if listing.in_stock:
                boost += 0.05
            raw.append((i, lex, vec, boost))

        if not raw:
            return []
        max_lex = max(r[1] for r in raw) or 1.0
        max_vec = max(r[2] for r in raw) or 1.0
        hits = []
        for i, lex, vec, boost in raw:
            blended = self.alpha * (lex / max_lex) + (1 - self.alpha) * (vec / max_vec) + boost
            hits.append(Hit(listing=self.listings[i], score=round(blended, 4),
                            lexical=round(lex, 4), vector=round(vec, 4), boost=boost))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # -- persistence ------------------------------------------------------
    def state(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "alpha": self.alpha,
            "vectors": {l.listing_id: v for l, v in zip(self.listings, self.vectors)},
        }


def _passes_filters(listing: Listing, spec: QuerySpec) -> bool:
    if spec.brands and listing.brand not in spec.brands:
        return False
    if spec.in_stock_only and not listing.in_stock:
        return False
    if spec.size and not listing.has_size(spec.size):
        return False
    if spec.gender and listing.gender and listing.gender not in (spec.gender, "unisex"):
        return False
    if spec.max_price is not None:
        price = listing.total_price if spec.include_shipping else listing.price
        if price > spec.max_price:
            return False
    if spec.min_discount is not None and listing.discount_pct < spec.min_discount:
        return False
    return True

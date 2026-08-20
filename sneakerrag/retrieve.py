"""Hybrid retrieval: field-weighted BM25 over an inverted index, fused with
vector similarity by reciprocal rank.

Three things make this different from a textbook top-k vector search, and each
exists because of how this data actually behaves:

1. **An inverted index, not a scan.** Only documents containing a query term
   are scored, and every per-document statistic (term frequencies, lengths) is
   computed once at build time instead of per query.
2. **Fields carry different weight.** A term matching a style code or the model
   name means far more than the same term buried in a marketing title, so the
   index is BM25F-style: term frequencies are weighted per field.
3. **Reciprocal-rank fusion, not a weighted sum of scores.** BM25 and cosine
   live on incomparable scales; normalising by the max lets one outlier squash
   the rest of the field. RRF combines the two *rankings*, which is scale-free
   and far more stable.

Structured filters (brand, size, condition, stock, budget) are applied as hard
predicates before scoring — a shopping constraint is a yes/no, not a nudge.
"""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .aliases import (
    FUZZY_WEIGHT,
    expand_query,
    meaningful_tokens,
    similarity,
    term_sources,
    trigrams,
)
from .embeddings import Embedder, cosine, get_embedder
from .models import Listing, QuerySpec
from .normalize import COLOR_WORDS, normalize_style_code, squash

try:  # optional: turns the cosine pass into one matrix multiply
    import numpy as _np
except Exception:  # pragma: no cover - numpy is not required
    _np = None

K1 = 1.5
B = 0.75
RRF_K = 60               # standard reciprocal-rank-fusion damping constant
DEFAULT_ALPHA = 0.6      # relative weight of the lexical ranking in the fusion
FUZZY_THRESHOLD = 0.45   # trigram similarity needed to treat a term as a typo
CODE_BOOST = 1.0         # an exact style-code hit outranks everything else

# Relevance floor. Without one, a query for a shoe the catalogue does not carry
# ("asics gel kayano 31") still returns its nearest neighbour, and the agent
# then prices a shoe nobody asked about — a fluent answer about the wrong
# product, which is worse than no answer.
MIN_TERM_COVERAGE = 0.4  # share of typed words a listing must satisfy ...
HIGH_COVERAGE = 0.8      # ... or nearly all of them, which stands on its own
MIN_VECTOR = 0.55        # ... unless it is very similar overall

# Words that describe a shoe without identifying one. Matching only these is
# not evidence: every catalogue has something black, low and retro, so a query
# for a shoe we do not carry would otherwise always find a "close enough" one.
GENERIC_TERMS = COLOR_WORDS | {
    "og", "low", "mid", "high", "retro", "classic", "vintage", "premium",
    "new", "edition", "grade", "school", "big", "little", "men", "mens",
    "women", "womens", "unisex", "kids", "deadstock", "used",
    # Brands are a hard filter, not an identity: "New Balance 2002R" matching a
    # 990v6 on the words "new balance" is the same mistake as matching on
    # "black". The model is what identifies the shoe.
    "nike", "adidas", "jordan", "balance", "nb",
    # "Air" spans Air Max, Air Force and Air Jordan — it places a shoe in a
    # family, it does not pick one out of it.
    "air",
}

# Two-stage retrieval: BM25 is a cheap first pass, and only its best candidates
# are worth the cosine and fusion work. Bounds query cost by the depth rather
# than by catalogue size.
RERANK_DEPTH = 400
COMMON_TERM_DF = 0.5     # a term in over half the corpus discriminates nothing
RARE_TERM_DF = 0.1       # ... and can be skipped when a sharper term is present
POSTING_BUDGET = 20_000  # postings visited before common terms switch to refining

# BM25F field weights: where a term matched matters as much as how often.
FIELD_WEIGHTS: dict[str, float] = {
    "style_code": 4.0,
    "model": 3.0,
    "brand": 2.0,
    "colorway": 2.0,
    "gender": 1.0,
    "source_name": 1.0,
    "title": 1.0,
}


_VARIANT = re.compile(r"^(\d{2,4})(v\d{1,2})$")
VARIANT_WEIGHT = 0.5     # a sub-token is weaker evidence than the whole word


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9.]+", squash(text)) if t]


def variants(token: str) -> list[str]:
    """Sub-tokens a shopper might type instead of the full model name.

    "990v6" is one token, so a search for "990" would otherwise miss it
    entirely — and "990" is what people actually type.
    """
    match = _VARIANT.match(token)
    return [match.group(1), match.group(2)] if match else []


@dataclass
class Hit:
    listing: Listing
    score: float
    lexical: float = 0.0
    vector: float = 0.0
    boost: float = 0.0
    terms: tuple[str, ...] = ()      # query terms this listing actually matched


class VectorIndex:
    """In-memory hybrid index over listings.

    Deliberately in-process: a few thousand listings is the realistic size of a
    price-comparison catalogue for three brands, and an approximate-nearest-
    neighbour service would add a dependency, a build step and recall loss to
    solve a problem this corpus does not have. The seams to swap in one are
    :class:`~sneakerrag.embeddings.Embedder` and :meth:`_vector_scores`.
    """

    def __init__(self, embedder: Embedder | None = None, alpha: float = DEFAULT_ALPHA,
                 fusion: str = "hybrid", expand: bool = True,
                 full_vector_scan: bool | None = None) -> None:
        self.embedder = embedder or get_embedder()
        # The default embedder is a hashed bag of n-grams: it is a fuzzy
        # *lexical* signal, so scanning the whole corpus with it adds cost
        # without adding recall, and it reranks the lexical pool instead. A real
        # dense model earns the full scan, so it gets one automatically.
        self.full_vector_scan = (full_vector_scan if full_vector_scan is not None
                                 else not self.embedder.name.startswith("hashing"))
        self.alpha = alpha
        self.fusion = fusion              # "hybrid" | "linear" | "rrf"
        self.expand = expand
        self.listings: list[Listing] = []
        self.vectors: list[list[float]] = []
        self.term_freqs: list[dict[str, float]] = []   # weighted tf per document
        self.doc_len: list[float] = []
        self.postings: dict[str, list[int]] = {}
        self.df: dict[str, int] = {}
        self.trigram_index: dict[str, set[str]] = {}
        self.avg_len: float = 0.0
        self._indexed_upto = 0            # append-only cursor into term_freqs
        self._matrix = None               # numpy view of self.vectors, when available

    # -- build ------------------------------------------------------------
    @staticmethod
    def _fields(listing: Listing) -> dict[str, str]:
        return {
            "style_code": listing.style_code,
            "model": listing.model,
            "brand": listing.brand,
            "colorway": listing.colorway,
            "gender": listing.gender,
            "source_name": listing.source_name,
            "title": listing.title,
        }

    def _document(self, listing: Listing) -> tuple[dict[str, float], float]:
        """Weighted term frequencies and effective length for one listing."""
        freqs: dict[str, float] = {}
        length = 0.0
        for field, text in self._fields(listing).items():
            weight = FIELD_WEIGHTS.get(field, 1.0)
            tokens = tokenize(text)
            for token in tokens:
                freqs[token] = freqs.get(token, 0.0) + weight
                for part in variants(token):
                    freqs[part] = freqs.get(part, 0.0) + weight * VARIANT_WEIGHT
            length += weight * len(tokens)
        return freqs, length

    def add(self, listings: Iterable[Listing],
            vectors: Sequence[Sequence[float]] | None = None) -> None:
        items = list(listings)
        if not items:
            return
        vecs = list(vectors) if vectors else self.embedder.embed_many([l.summary() for l in items])
        for listing, vec in zip(items, vecs):
            freqs, length = self._document(listing)
            self.listings.append(listing)
            self.vectors.append(list(vec))
            self.term_freqs.append(freqs)
            self.doc_len.append(length)
        self._reindex()

    def _reindex(self) -> None:
        """Index the documents added since the last call (append-only)."""
        for doc_id in range(self._indexed_upto, len(self.term_freqs)):
            for term in self.term_freqs[doc_id]:
                if term not in self.postings:
                    self.postings[term] = []
                    for gram in trigrams(term):
                        self.trigram_index.setdefault(gram, set()).add(term)
                self.postings[term].append(doc_id)
                self.df[term] = self.df.get(term, 0) + 1
        self._indexed_upto = len(self.term_freqs)
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self._matrix = (_np.asarray(self.vectors, dtype="float32")
                        if _np is not None and self.vectors else None)

    def __len__(self) -> int:
        return len(self.listings)

    # -- query terms ------------------------------------------------------
    def resolve_terms(self, query: str) -> list[tuple[str, float]]:
        """Expand a query, then repair terms that exist nowhere in the corpus."""
        raw = expand_query(query) if self.expand else [(t, 1.0) for t in tokenize(query)]
        resolved: list[tuple[str, float]] = []
        for term, weight in raw:
            if term in self.postings:
                resolved.append((term, weight))
                continue
            corrected = self._nearest_term(term)
            if corrected:
                resolved.append((corrected, weight * FUZZY_WEIGHT))
        return resolved

    def _nearest_term(self, term: str) -> str:
        """Closest indexed term by trigram overlap — cheap typo tolerance."""
        if len(term) < 4:
            return ""
        candidates: set[str] = set()
        for gram in trigrams(term):
            candidates |= self.trigram_index.get(gram, set())
        best, best_score = "", FUZZY_THRESHOLD
        for candidate in candidates:
            score = similarity(term, candidate)
            if score > best_score:
                best, best_score = candidate, score
        return best

    # -- scoring ----------------------------------------------------------
    def _lexical_scores(self, terms: Sequence[tuple[str, float]],
                        eligible: set[int]) -> tuple[dict[int, float], dict[int, set[str]]]:
        """BM25F over the inverted index, rarest term first.

        Rare terms select the candidates; common ones ("white", "low") only
        refine the ranking among them. Once the posting budget is spent, a
        common term stops walking its whole list and just scores documents that
        earlier terms already surfaced — which is where the time goes on a large
        catalogue, and costs nothing in quality because a listing matching only
        colour words fails the relevance floor anyway.
        """
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        n = len(self.listings)
        useful = self._discriminative(terms, n)
        budget = POSTING_BUDGET

        def score_doc(doc: int, term: str, weight: float, idf: float) -> None:
            freq = self.term_freqs[doc].get(term, 0.0)
            if not freq:
                return
            denom = freq + K1 * (1 - B + B * self.doc_len[doc] / (self.avg_len or 1))
            scores[doc] = scores.get(doc, 0.0) + weight * idf * (freq * (K1 + 1)) / denom
            matched.setdefault(doc, set()).add(term)

        for term, weight in sorted(useful, key=lambda tw: self.df.get(tw[0], 0)):
            docs = self.postings.get(term)
            if not docs:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            if scores and len(docs) > budget:
                for doc in list(scores):
                    score_doc(doc, term, weight, idf)
                continue
            budget -= len(docs)
            for doc in docs:
                if doc in eligible:
                    score_doc(doc, term, weight, idf)
        return scores, matched

    def _discriminative(self, terms: Sequence[tuple[str, float]],
                        n: int) -> list[tuple[str, float]]:
        """Drop terms that match most of the catalogue when a sharp term exists.

        Scoring "black" against 30,000 listings costs real time and moves
        nothing; if the same query also contains "ultraboost", that term decides
        the result on its own.
        """
        if not terms or n < 100:
            return list(terms)
        dfs = {term: self.df.get(term, 0) for term, _weight in terms}
        if not any(0 < df <= RARE_TERM_DF * n for df in dfs.values()):
            return list(terms)
        return [(term, weight) for term, weight in terms
                if dfs.get(term, 0) <= COMMON_TERM_DF * n]

    def _vector_scores(self, query: str, eligible: Sequence[int]) -> dict[int, float]:
        if not query.strip() or not self.vectors:
            return {}
        q = self.embedder.embed(query)
        if self._matrix is not None:
            qv = _np.asarray(q, dtype="float32")
            # Gathering rows copies the sub-matrix, which costs more than the
            # multiply itself once most of the corpus is eligible.
            if len(eligible) * 3 >= len(self.listings):
                sims = self._matrix @ qv
                return {doc: float(sims[doc]) for doc in eligible}
            rows = _np.fromiter(eligible, dtype="int64", count=len(eligible))
            sims = self._matrix[rows] @ qv
            return {int(doc): float(score) for doc, score in zip(rows, sims)}
        return {doc: cosine(q, self.vectors[doc]) for doc in eligible}

    def _fuse(self, candidates: set[int], lexical: dict[int, float],
              vector: dict[int, float]) -> dict[int, float]:
        """Combine the two signals.

        ``linear`` normalises each score by its maximum; ``rrf`` uses reciprocal
        ranks only. Measured on the golden set, neither dominates: the linear
        blend is sharper at rank 1 because BM25's *margin* carries real signal
        (a style-code hit is not merely "first", it is far ahead), while RRF is
        steadier deeper in the list because it cannot be skewed by one outlier
        score. ``hybrid`` — the default — averages the two, and beats both.
        """
        max_lex = max(lexical.values(), default=0.0) or 1.0
        max_vec = max(vector.values(), default=0.0) or 1.0
        lex_rank = self._ranks(lexical)
        vec_rank = self._ranks(vector)

        def linear(doc: int) -> float:
            return (self.alpha * (lexical.get(doc, 0.0) / max_lex)
                    + (1 - self.alpha) * (vector.get(doc, 0.0) / max_vec))

        def rrf(doc: int) -> float:
            score = 0.0
            if doc in lex_rank:
                score += self.alpha / (RRF_K + lex_rank[doc])
            if doc in vec_rank:
                score += (1 - self.alpha) / (RRF_K + vec_rank[doc])
            # Scale onto roughly the same 0..1 range as the linear half so the
            # hybrid average is not dominated by one term.
            return score * (RRF_K + 1)

        if self.fusion == "linear":
            return {doc: linear(doc) for doc in candidates}
        if self.fusion == "rrf":
            return {doc: rrf(doc) for doc in candidates}
        return {doc: 0.5 * (linear(doc) + rrf(doc)) for doc in candidates}

    @staticmethod
    def _relevant(doc: int, matched: dict[int, set[str]], vector: dict[int, float],
                  wanted: set[str], sources: dict[str, set[str]]) -> bool:
        """Is this listing plausibly what was asked for, or just the least-bad
        thing in the catalogue?

        Two conditions, both required. Enough of the *typed* words must be
        satisfied — credited back through any expansion, so a nickname cannot
        vouch for itself — and at least one of them must actually identify a
        shoe rather than describe one. Asking for an "Air Jordan 1 Retro High OG
        Chicago" should not return a Dunk Low Retro on the strength of "retro"
        plus the colours "Chicago" happens to expand to.
        """
        hit_terms = matched.get(doc, set())
        satisfied = {source
                     for term in hit_terms
                     for source in sources.get(term, {term})} & wanted
        coverage = len(satisfied) / len(wanted)
        identifying = any(term not in GENERIC_TERMS for term in hit_terms)
        # Either something identified the model, or essentially everything the
        # shopper typed was found — a pure-colorway query ("silver sea salt new
        # balance") is legitimate precisely because it matches in full.
        strong = coverage >= HIGH_COVERAGE or (identifying and coverage >= MIN_TERM_COVERAGE)
        return strong or vector.get(doc, 0.0) >= MIN_VECTOR

    @staticmethod
    def _ranks(scores: dict[int, float]) -> dict[int, int]:
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return {doc: rank for rank, (doc, _score) in enumerate(ordered, start=1)}

    def search(self, query: str, spec: QuerySpec | None = None, top_k: int = 40) -> list[Hit]:
        spec = spec or QuerySpec(text=query)
        if not self.listings:
            return []

        eligible = [i for i, listing in enumerate(self.listings)
                    if _passes_filters(listing, spec)]
        if not eligible:
            return []
        eligible_set = set(eligible)

        terms = self.resolve_terms(query)
        lexical, matched = self._lexical_scores(terms, eligible_set)
        if len(lexical) > RERANK_DEPTH:
            # Second stage only sees the strongest first-stage candidates.
            keep = set(heapq.nlargest(RERANK_DEPTH, lexical, key=lexical.get))
            lexical = {doc: score for doc, score in lexical.items() if doc in keep}
        # No lexical match at all means the query is pure paraphrase — then the
        # vector pass has to cover the whole eligible set to find anything.
        scan = eligible if (self.full_vector_scan or not lexical) else sorted(lexical)
        vector = self._vector_scores(query, scan)

        code = normalize_style_code(spec.style_code)
        candidates = set(lexical) | set(vector)
        if not candidates:
            candidates = eligible_set

        fused = self._fuse(candidates, lexical, vector)

        # Coverage is measured against what the shopper actually typed, not
        # against the terms that survived resolution — a query whose words are
        # absent from the catalogue must score zero, not be judged on the one
        # word that happened to match.
        wanted = set(meaningful_tokens(query)) if self.expand else {t for t, _w in terms}
        sources = term_sources(query) if self.expand else {}
        hits: list[Hit] = []
        for doc, score in fused.items():
            listing = self.listings[doc]
            boost = CODE_BOOST if code and normalize_style_code(listing.style_code) == code else 0.0
            # An exact style code is an identity, not evidence — it outranks the
            # relevance floor rather than being judged by it.
            if not boost and wanted and not self._relevant(doc, matched, vector, wanted, sources):
                continue
            hits.append(Hit(listing=listing, score=round(score + boost, 6),
                            lexical=round(lexical.get(doc, 0.0), 4),
                            vector=round(vector.get(doc, 0.0), 4),
                            boost=boost, terms=tuple(sorted(matched.get(doc, ())))))
        hits.sort(key=lambda h: (-h.score, h.listing.total_price))
        return hits[:top_k]

    # -- persistence ------------------------------------------------------
    def state(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "alpha": self.alpha,
            "fusion": self.fusion,
            "vectors": {l.listing_id: v for l, v in zip(self.listings, self.vectors)},
        }


def _passes_filters(listing: Listing, spec: QuerySpec) -> bool:
    if spec.brands and listing.brand not in spec.brands:
        return False
    if spec.in_stock_only and not listing.in_stock:
        return False
    if spec.size and not listing.has_size(spec.size):
        return False
    if spec.condition and listing.condition != spec.condition:
        return False
    if spec.gender and listing.gender and listing.gender not in (spec.gender, "unisex"):
        return False
    if spec.max_price is not None:
        # Budget applies to the requested size: on a resale marketplace the
        # headline "from" price is usually a size nobody asked for.
        price = (listing.total_for(spec.size) if spec.include_shipping
                 else listing.price_for(spec.size))
        if price > spec.max_price:
            return False
    if spec.min_discount is not None and listing.discount_pct_for(spec.size) < spec.min_discount:
        return False
    return True

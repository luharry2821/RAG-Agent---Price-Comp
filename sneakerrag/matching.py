"""Group retailer listings into canonical products (the "same SKU" problem).

Comparing prices is only meaningful once we are confident two listings are the
same physical shoe. We use a two-tier strategy:

1. **Style code join** — a manufacturer code (``DD1391-100``, ``B75806``,
   ``M990GL6``) is authoritative. Same code => same shoe, full stop.
2. **Fuzzy join** — for the many listings that never expose a code, compare
   brand + model + colorway + gender with a similarity score, guarded by hard
   negative rules (different model numbers or conflicting codes never merge).

The result is a set of :class:`~sneakerrag.models.Product` clusters, each of
which can be priced across sources.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from .models import Listing, Product
from .normalize import normalize_colorway, normalize_style_code, slug, squash

MODEL_THRESHOLD = 0.82
COLORWAY_THRESHOLD = 0.45

_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")
_VERSION = re.compile(r"\bv(\d+)\b")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in self.parent:
            out.setdefault(self.find(node), []).append(node)
        return out


# --------------------------------------------------------------------------
# Similarity primitives
# --------------------------------------------------------------------------

def token_set(text: str) -> set[str]:
    return {t for t in re.split(r"[\s/_-]+", squash(text)) if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_ratio(a: str, b: str) -> float:
    a2 = " ".join(sorted(token_set(a)))
    b2 = " ".join(sorted(token_set(b)))
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def text_similarity(a: str, b: str) -> float:
    """Blend of token overlap and character-level similarity, 0..1."""
    if not a or not b:
        return 0.0
    if squash(a) == squash(b):
        return 1.0
    return max(jaccard(token_set(a), token_set(b)), sequence_ratio(a, b))


def model_numbers(model: str) -> set[str]:
    """Digits that identify the silhouette: 990, 574, 90, 95, plus vN."""
    nums = set(_NUM_TOKEN.findall(squash(model)))
    nums |= {f"v{m}" for m in _VERSION.findall(squash(model))}
    return nums


def model_similarity(a: str, b: str) -> float:
    """Model similarity with a hard gate on differing model numbers.

    "990v6" and "990v5" are different shoes even though the strings are 95%
    identical, so a mismatch in the numeric part zeroes the score.
    """
    na, nb = model_numbers(a), model_numbers(b)
    if na and nb and na != nb:
        return 0.0
    base = text_similarity(a, b)
    # "990v6" vs "990v6 made in usa": identical model numbers and one title is
    # a strict superset of the other, so the extra words are qualifiers.
    ta, tb = token_set(a), token_set(b)
    if na and na == nb and (ta <= tb or tb <= ta):
        return max(base, 0.9)
    return base


def colorway_similarity(a: str, b: str) -> float:
    a, b = normalize_colorway(a), normalize_colorway(b)
    if not a or not b:
        return 0.6            # unknown colour: neither evidence for nor against
    return text_similarity(a, b)


def gender_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    if "unisex" in (a, b):
        return True
    return a == b


# --------------------------------------------------------------------------
# Pairwise decision
# --------------------------------------------------------------------------

def match_score(a: Listing, b: Listing) -> tuple[float, str]:
    """Return (confidence, reason) that two listings are the same shoe."""
    if a.brand and b.brand and a.brand != b.brand:
        return 0.0, "different brand"

    code_a = normalize_style_code(a.style_code)
    code_b = normalize_style_code(b.style_code)
    if code_a and code_b:
        if code_a == code_b:
            return 1.0, f"style code {code_a}"
        return 0.0, f"conflicting style codes {code_a} vs {code_b}"

    if not gender_compatible(a.gender, b.gender):
        return 0.0, f"different audience ({a.gender} vs {b.gender})"

    m = model_similarity(a.model or a.title, b.model or b.title)
    if m < MODEL_THRESHOLD:
        return 0.0, f"model similarity {m:.2f} < {MODEL_THRESHOLD}"

    c = colorway_similarity(a.colorway, b.colorway)
    if c < COLORWAY_THRESHOLD:
        return 0.0, f"colorway similarity {c:.2f} < {COLORWAY_THRESHOLD}"

    confidence = round(0.65 * m + 0.35 * c, 3)
    return confidence, f"fuzzy match (model {m:.2f}, colorway {c:.2f})"


def is_same_product(a: Listing, b: Listing, min_confidence: float = 0.78) -> bool:
    return match_score(a, b)[0] >= min_confidence


def explain(a: Listing, b: Listing) -> str:
    score, reason = match_score(a, b)
    verdict = "MATCH" if score >= 0.78 else "no match"
    return f"{verdict} ({score:.2f}): {reason}"


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def _blocking_keys(listing: Listing) -> set[str]:
    """Cheap keys that co-locate candidates so we avoid O(n^2) comparisons."""
    brand = listing.brand or "?"
    keys = set()
    model = listing.model or listing.title
    tokens = [t for t in token_set(model) if t]
    for t in tokens[:3]:
        keys.add(f"{brand}|{t}")
    for n in model_numbers(model):
        keys.add(f"{brand}|#{n}")
    if not keys:
        keys.add(brand)
    return keys


def _cluster_codes(members: Sequence[Listing]) -> set[str]:
    return {normalize_style_code(m.style_code) for m in members if m.style_code}


def _cluster_colorways(members: Sequence[Listing]) -> set[str]:
    return {m.colorway for m in members if m.colorway}


def clusters_compatible(a: Sequence[Listing], b: Sequence[Listing],
                        min_confidence: float = 0.78) -> bool:
    """Decide whether two *clusters* describe the same shoe.

    Comparing clusters rather than bare listings is what stops a vague listing
    from bridging two real products: a title with no colorway can look like a
    match for "Air Max 90 White/Black" *and* "Air Max 90 Infrared", but once it
    has joined one cluster, that cluster's known colorway blocks the other.
    """
    codes_a, codes_b = _cluster_codes(a), _cluster_codes(b)
    if codes_a and codes_b:
        return bool(codes_a & codes_b)

    brands_a = {m.brand for m in a if m.brand}
    brands_b = {m.brand for m in b if m.brand}
    if brands_a and brands_b and not (brands_a & brands_b):
        return False

    for x in a:
        for y in b:
            if not gender_compatible(x.gender, y.gender):
                return False
            if model_similarity(x.model or x.title, y.model or y.title) == 0.0:
                return False        # conflicting model numbers anywhere = veto

    best_model = max(model_similarity(x.model or x.title, y.model or y.title)
                     for x in a for y in b)
    if best_model < MODEL_THRESHOLD:
        return False

    colors_a, colors_b = _cluster_colorways(a), _cluster_colorways(b)
    if colors_a and colors_b:
        best_color = max(colorway_similarity(x, y) for x in colors_a for y in colors_b)
    else:
        best_color = 0.6            # nothing known either way
    if best_color < COLORWAY_THRESHOLD:
        return False

    return round(0.65 * best_model + 0.35 * best_color, 3) >= min_confidence


def _ambiguous_merge(ra: str, rb: str, members: dict[str, list[Listing]],
                     competing_colorways) -> bool:
    """True when a colourless cluster has several incompatible suitors."""
    colors_a = _cluster_colorways(members[ra])
    colors_b = _cluster_colorways(members[rb])
    if bool(colors_a) == bool(colors_b):
        return False                       # both known or both unknown: no ambiguity
    unknown_root = ra if not colors_a else rb
    distinct: list[set[str]] = []
    for colors in competing_colorways(unknown_root):
        if not any(max(colorway_similarity(x, y) for x in colors for y in seen)
                   >= COLORWAY_THRESHOLD for seen in distinct):
            distinct.append(colors)
    return len(distinct) > 1


def cluster_listings(listings: Iterable[Listing],
                     min_confidence: float = 0.78) -> list[Product]:
    """Group listings into products, cheapest-first inside each product.

    Two passes: an exact style-code join, then a greedy agglomerative pass over
    the resulting clusters, best-scoring candidate pair first.
    """
    items = list(listings)
    if not items:
        return []
    uf = UnionFind()
    for l in items:
        uf.add(l.listing_id)

    # Pass 1: exact style-code join across the whole corpus.
    by_code: dict[str, list[Listing]] = {}
    for l in items:
        code = normalize_style_code(l.style_code)
        if code:
            by_code.setdefault(code, []).append(l)
    for group in by_code.values():
        # Style codes are effectively globally unique, but if two brands somehow
        # share a code string, keep them apart rather than merge blindly.
        brands = {l.brand for l in group if l.brand}
        if len(brands) > 1:
            continue
        for other in group[1:]:
            uf.union(group[0].listing_id, other.listing_id)

    # Pass 2: score blocked candidate pairs, then merge greedily. Each merge is
    # re-validated against the *current* clusters, so an early merge can veto a
    # later one that would have been wrong.
    blocks: dict[str, list[Listing]] = {}
    for l in items:
        for key in _blocking_keys(l):
            blocks.setdefault(key, []).append(l)

    scored: list[tuple[float, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for group in blocks.values():
        if len(group) > 400:              # pathological block; style codes carry it
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pair = tuple(sorted((a.listing_id, b.listing_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                score = match_score(a, b)[0]
                if score >= min_confidence:
                    scored.append((score, pair[0], pair[1]))

    members: dict[str, list[Listing]] = {}
    for l in items:
        members.setdefault(uf.find(l.listing_id), []).append(l)

    adjacency: dict[str, set[str]] = {}
    for _score, id_a, id_b in scored:
        adjacency.setdefault(id_a, set()).add(id_b)
        adjacency.setdefault(id_b, set()).add(id_a)

    def competing_colorways(root: str) -> list[set[str]]:
        """Colorways of every other cluster this one could plausibly join."""
        out: list[set[str]] = []
        for member in members.get(root, []):
            for other_id in adjacency.get(member.listing_id, ()):
                other_root = uf.find(other_id)
                if other_root == root:
                    continue
                colors = _cluster_colorways(members.get(other_root, []))
                if colors and colors not in out:
                    out.append(colors)
        return out

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    for _score, id_a, id_b in scored:
        ra, rb = uf.find(id_a), uf.find(id_b)
        if ra == rb:
            continue
        if not clusters_compatible(members[ra], members[rb], min_confidence):
            continue
        if _ambiguous_merge(ra, rb, members, competing_colorways):
            # This listing could belong to two different colorways and we have
            # no evidence to choose. Leave it out rather than quote a price for
            # the wrong shoe.
            continue
        uf.union(ra, rb)
        root = uf.find(ra)
        merged = members.pop(ra) + members.pop(rb)
        members[root] = merged

    products = [_build_product(group) for group in members.values() if group]
    products.sort(key=lambda p: (p.brand, p.model, p.colorway))
    return products


def _build_product(members: list[Listing]) -> Product:
    """Pick the most complete metadata across a cluster's listings."""
    def vote(attr: str) -> str:
        counts: dict[str, int] = {}
        for l in members:
            v = (getattr(l, attr) or "").strip()
            if v:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            return ""
        # Most common wins; ties broken by longest (most descriptive) value.
        return sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[-1][0]

    brand = vote("brand")
    model = vote("model")
    colorway = vote("colorway")
    gender = vote("gender")
    code = normalize_style_code(vote("style_code"))

    if code:
        key = f"{brand}:{code}".replace(" ", "-").lower()
    else:
        key = ":".join(x for x in (slug(brand), slug(model), slug(colorway), gender) if x)

    ordered = sorted(members, key=lambda l: (not l.in_stock, l.total_price))
    return Product(product_key=key, brand=brand, model=model, colorway=colorway,
                   style_code=code, gender=gender, listings=ordered)

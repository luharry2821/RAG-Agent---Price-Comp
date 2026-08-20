#!/usr/bin/env python3
"""Measure retrieval speed and quality.

Speed is timed against a synthetic catalogue at several sizes; quality is
scored on a golden set of real-shopper phrasings against the bundled fixtures.
Both halves can be run with the retrieval knobs flipped, so an improvement is
demonstrated rather than asserted.

    python3 tools/benchmark.py            # both
    python3 tools/benchmark.py --speed
    python3 tools/benchmark.py --quality
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sneakerrag.matching import cluster_listings
from sneakerrag.models import Listing, QuerySpec
from sneakerrag.retrieve import VectorIndex
from sneakerrag.sources import get_sources

# A shopper's phrasing -> the style code they meant. Nicknames, plurals,
# misspellings and colorway slang are all deliberately represented.
GOLDEN: list[tuple[str, str]] = [
    ("cheapest air max 90", "HM0089-100"),
    ("am90 white black", "HM0089-100"),
    ("nike air max 90 in a 10.5", "HM0089-100"),
    ("af1", "CW2288-111"),
    ("af1s triple white", "CW2288-111"),
    ("air force 1 07", "CW2288-111"),
    ("panda dunks", "DD1391-100"),
    ("dunk low retro white black", "DD1391-100"),
    ("nike dunks size 13", "DD1391-100"),
    ("pegasus 41", "FD2723-001"),
    ("sambas", "B75806"),
    ("adiddas samba og", "B75806"),
    ("samba cloud white core black", "B75806"),
    ("ub 5", "IE1766"),
    ("ultrabost core black", "IE1766"),
    ("adidas ultraboost 5", "IE1766"),
    ("campus 00s grey", "HQ8708"),
    ("nb 990s", "M990GL6"),
    ("new balence 990v6", "M990GL6"),
    ("990v6 grey", "M990GL6"),
    ("new balance 550 white green", "BB550PB1"),
    ("550s", "BB550PB1"),
    ("1906r silver", "M1906RA"),
    ("dad shoes", "M1906RA"),
    # Harder: colour-led, misspelled, code-led and gendered phrasings.
    ("black ultraboost", "IE1766"),
    ("white green 550", "BB550PB1"),
    ("grey campus 00s", "HQ8708"),
    ("womens pegasus", "FD2723-001"),
    ("peagasus 41 black white", "FD2723-001"),
    ("B75806", "B75806"),
    ("M990GL6 grey", "M990GL6"),
    ("af 1 white", "CW2288-111"),
    ("silver sea salt new balance", "M1906RA"),
    ("nike dunks size 13", "DD1391-100"),
]

# Shoes the catalogue does not carry. The right answer is *no* answer: returning
# the nearest neighbour means the agent prices a shoe nobody asked for.
NEGATIVES: list[str] = [
    "asics gel kayano 31",
    "jordan 4 bred",
    "puma speedcat",
    "on cloudmonster",
    "hoka clifton 9",
    "converse chuck 70",
]

MODELS = ["air max 90", "air max 95", "air max 97", "air force 1", "dunk low", "dunk high",
          "pegasus 41", "vomero 18", "samba og", "gazelle", "campus 00s", "ultraboost 5",
          "forum low", "stan smith", "990v6", "990v5", "550", "574", "1906r", "2002r",
          "fresh foam 1080v13", "9060", "530", "327"]
COLORS = ["white black", "triple white", "core black", "grey silver", "navy gum",
          "sea salt", "cloud white", "red white", "olive sail", "pink foam"]
BRANDS = {"air": "nike", "dunk": "nike", "pegasus": "nike", "vomero": "nike",
          "samba": "adidas", "gazelle": "adidas", "campus": "adidas",
          "ultraboost": "adidas", "forum": "adidas", "stan": "adidas"}


def synthetic_corpus(n: int, seed: int = 7) -> list[Listing]:
    """A catalogue with realistic vocabulary spread — most queries should touch
    a small slice of it, which is what an inverted index is for."""
    rng = random.Random(seed)
    sites = ["nike", "adidas", "newbalance", "stadiumgoods", "flightclub", "goat", "kickscrew"]
    out: list[Listing] = []
    for i in range(n):
        model = rng.choice(MODELS)
        color = rng.choice(COLORS)
        brand = next((b for key, b in BRANDS.items() if key in model), "new balance")
        out.append(Listing(
            source=rng.choice(sites), source_name="Site", url=f"https://example.test/p/{i}",
            title=f"{brand.title()} {model} {color}", brand=brand, model=model,
            colorway=color, style_code=f"AB{1000 + i % 9000}-{i % 900:03d}",
            price=round(rng.uniform(80, 260), 2), sizes=["9", "10", "11"]))
    return out


def bench_speed(sizes=(1_000, 10_000, 50_000), repeats: int = 25) -> None:
    queries = ["air max 90 white black", "samba og cloud white", "990v6 grey",
               "panda dunks", "ultraboost core black"]
    print(f"{'corpus':>8} {'build':>9} {'cached build':>13} {'query p50':>11} {'query p95':>11}")
    for n in sizes:
        listings = synthetic_corpus(n)
        index = VectorIndex()
        start = time.perf_counter()
        index.add(listings)
        build = time.perf_counter() - start

        # Re-adding with the cached vectors is what `ingest` does on a rerun.
        cached = VectorIndex()
        start = time.perf_counter()
        cached.add(listings, index.vectors)
        build_cached = time.perf_counter() - start

        timings = []
        for i in range(repeats):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            index.search(query, QuerySpec(in_stock_only=False), top_k=40)
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        p50 = timings[len(timings) // 2]
        p95 = timings[int(len(timings) * 0.95) - 1]
        print(f"{n:>8} {build:>8.2f}s {build_cached:>12.2f}s {p50:>10.1f}ms {p95:>10.1f}ms")


def _index_from_fixtures(**kwargs) -> tuple[VectorIndex, dict[str, str]]:
    listings = [l for adapter in get_sources() for l in adapter.search("", limit=999)]
    index = VectorIndex(**kwargs)
    index.add(listings)
    return index, {l.listing_id: l.style_code for l in listings}


def bench_quality(configs=None) -> None:
    configs = configs or [
        ("linear blend, no vocabulary", {"fusion": "linear", "expand": False}),
        ("linear blend + vocabulary", {"fusion": "linear", "expand": True}),
        ("RRF + vocabulary", {"fusion": "rrf", "expand": True}),
        ("hybrid + vocabulary (shipped)", {"fusion": "hybrid", "expand": True}),
    ]
    print(f"{'configuration':<30} {'recall@1':>9} {'recall@3':>9} {'MRR':>7} {'false hits':>11}  misses")
    for label, kwargs in configs:
        index, codes = _index_from_fixtures(**kwargs)
        hits_at_1 = hits_at_3 = 0
        reciprocal = 0.0
        misses = []
        for query, expected in GOLDEN:
            results = index.search(query, QuerySpec(in_stock_only=False), top_k=10)
            # Rank products, not listings: several listings share one style code.
            ranked: list[str] = []
            for hit in results:
                code = codes.get(hit.listing.listing_id, "")
                if code and code not in ranked:
                    ranked.append(code)
            position = ranked.index(expected) + 1 if expected in ranked else 0
            if position == 1:
                hits_at_1 += 1
            if 1 <= position <= 3:
                hits_at_3 += 1
            if position:
                reciprocal += 1 / position
            else:
                misses.append(query)
        # A false hit is any result at all for a shoe we do not stock.
        false_hits = sum(1 for query in NEGATIVES
                         if index.search(query, QuerySpec(in_stock_only=False), top_k=5))
        total = len(GOLDEN)
        print(f"{label:<30} {hits_at_1 / total:>8.0%} {hits_at_3 / total:>8.0%} "
              f"{reciprocal / total:>7.3f} {false_hits:>6}/{len(NEGATIVES):<4}  "
              f"{', '.join(misses[:3]) or '-'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--max-corpus", type=int, default=50_000)
    args = parser.parse_args()
    run_all = not (args.speed or args.quality)

    if args.quality or run_all:
        print(f"Retrieval quality — {len(GOLDEN)} shopper phrasings + "
              f"{len(NEGATIVES)} out-of-catalogue queries")
        print(f"({len(cluster_listings([l for a in get_sources() for l in a.search('', 999)]))} "
              f"products indexed)\n")
        bench_quality()
        print()
    if args.speed or run_all:
        print("Retrieval speed — synthetic catalogue")
        bench_speed(tuple(n for n in (1_000, 10_000, 50_000) if n <= args.max_corpus))


if __name__ == "__main__":
    main()

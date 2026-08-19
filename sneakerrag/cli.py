"""Command line interface: ingest, compare, ask, deals, product, stats, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SneakerAgent
from .matching import cluster_listings
from .models import fmt_money
from .normalize import normalize_style_code
from .sources import SITES, get_sources
from .store import Catalog, DEFAULT_DB

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def _color(enabled: bool):
    if enabled and sys.stdout.isatty():
        return BOLD, DIM, GREEN, YELLOW, RESET
    return "", "", "", "", ""


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_sources(args: argparse.Namespace) -> int:
    for spec in SITES:
        brands = ", ".join(spec.brands) if spec.brands else "nike, adidas, new balance"
        print(f"{spec.key:22} {spec.name:26} {brands}")
        if args.verbose:
            print(f"{'':22} {spec.home}\n{'':22} {spec.notes}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    keys = [k.strip() for k in args.sources.split(",") if k.strip()] if args.sources else None
    mode = "live" if args.live else "fixtures"
    adapters = get_sources(keys, mode=mode)
    queries = args.query or None
    with Catalog(args.db) as catalog:
        agent = SneakerAgent(catalog=catalog, use_llm=False)
        result = agent.ingest(adapters, queries=queries, limit=args.limit)
    print(f"mode: {mode}")
    for key, count in result["per_source"].items():
        print(f"  {key:22} {count:4d} listings")
    print(f"total {result['total']}  (new {result['new']}, updated {result['updated']}, "
          f"price changes {result['price_changed']})")
    return 0


def _print_products(products, spec, verbose: bool, use_color: bool) -> None:
    bold, dim, green, yellow, reset = _color(use_color)
    if not products:
        print("No matching products.")
        return
    for product in products:
        header = product.display_name
        if product.style_code:
            header += f"  [{product.style_code}]"
        print(f"\n{bold}{header}{reset}")
        offers = product.offers(size=spec.size, in_stock_only=spec.in_stock_only,
                                include_shipping=spec.include_shipping)
        excluded = {l.listing_id for l in product.listings} - {l.listing_id for l in offers}
        if not offers:
            print(f"  {yellow}no listing matches the requested size/stock filters{reset}")
        for rank, listing in enumerate(offers):
            marker = f"{green}CHEAPEST{reset}" if rank == 0 else "        "
            ship = ("free ship" if listing.shipping == 0 else
                    f"+{fmt_money(listing.shipping, listing.currency)}" if listing.shipping else "ship n/a")
            promo = f"  {listing.discount_pct:g}% off" if listing.discount_pct else ""
            print(f"  {marker} {listing.source_name:26} {fmt_money(listing.price, listing.currency):>10}"
                  f"  {ship:>10}  = {fmt_money(listing.total_price, listing.currency):>10}{promo}")
            if verbose:
                print(f"           {dim}{listing.url}{reset}")
                print(f"           {dim}sizes: {', '.join(listing.sizes) or 'n/a'}{reset}")
        if offers:
            saving = round(offers[-1].total_price - offers[0].total_price, 2)
            if saving > 0:
                print(f"  {dim}spread across {len(offers)} sites: {fmt_money(saving)}{reset}")
        for listing_id in excluded:
            listing = next(l for l in product.listings if l.listing_id == listing_id)
            why = "out of stock" if not listing.in_stock else f"no size {spec.size}"
            print(f"  {dim}excluded {listing.source_name} ({why}) "
                  f"{fmt_money(listing.price, listing.currency)}{reset}")


def cmd_compare(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    with Catalog(args.db) as catalog:
        agent = SneakerAgent(catalog=catalog, use_llm=False)
        spec, products = agent.compare(query, limit=args.limit)
    if args.json:
        print(json.dumps({"query": query, "spec": spec.to_dict(),
                          "products": [p.to_dict() for p in products]}, indent=2, default=str))
        return 0
    print(f"query: {query}\nfilters: {spec.describe()}")
    _print_products(products, spec, args.verbose, not args.no_color)
    return 0 if products else 1


def cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    with Catalog(args.db) as catalog:
        agent = SneakerAgent(catalog=catalog, use_llm=not args.no_llm)
        answer = agent.answer(question, limit=args.limit)
    if args.json:
        print(answer.to_json())
        return 0
    print(answer.text)
    if answer.citations:
        print("\nSources")
        for citation in answer.citations:
            print("  " + citation.render())
    print(f"\n({'Claude ' + agent.llm.name if answer.generator == 'claude' else 'offline template'};"
          f" {len(agent.index)} listings indexed)")
    return 0


def cmd_product(args: argparse.Namespace) -> int:
    code = normalize_style_code(args.style_code)
    with Catalog(args.db) as catalog:
        listings = [l for l in catalog.listings()
                    if normalize_style_code(l.style_code) == code]
        if not listings:
            print(f"No listings with style code {code}.")
            return 1
        product = cluster_listings(listings)[0]
        print(f"{product.display_name}  [{product.style_code}]")
        for listing in product.offers(in_stock_only=False):
            print(f"  {listing.source_name:26} {fmt_money(listing.price, listing.currency):>10}"
                  f"  delivered {fmt_money(listing.total_price, listing.currency):>10}"
                  f"  {'in stock' if listing.in_stock else 'OUT OF STOCK'}")
            print(f"    {listing.url}")
            history = catalog.price_history(listing.listing_id)
            if len(history) > 1:
                trail = " -> ".join(f"{h['price']:.2f}" for h in history[-5:])
                print(f"    price history: {trail}")
    return 0


def cmd_deals(args: argparse.Namespace) -> int:
    with Catalog(args.db) as catalog:
        rows = []
        for listing in catalog.listings(brand=args.brand):
            if listing.discount_pct >= args.min_pct and listing.in_stock:
                rows.append(listing)
        rows.sort(key=lambda l: -l.discount_pct)
        if not rows:
            print(f"No listings at {args.min_pct:g}% off or better.")
            return 1
        for listing in rows[: args.limit]:
            print(f"{listing.discount_pct:5.1f}%  {listing.source_name:26} "
                  f"{fmt_money(listing.price, listing.currency):>10} "
                  f"(was {fmt_money(listing.list_price, listing.currency)})  {listing.title}")
            print(f"        {listing.url}")

        drops = catalog.price_drops(min_pct=args.min_pct)
        if drops:
            print(f"\nPrice drops since first seen ({len(drops)}):")
            by_id = {l.listing_id: l for l in catalog.listings()}
            for drop in drops[:10]:
                listing = by_id.get(drop["listing_id"])
                if listing:
                    print(f"  -{drop['drop_pct']:.1f}%  {listing.source_name}: "
                          f"{fmt_money(drop['high'])} -> {fmt_money(drop['now'])}  {listing.title}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with Catalog(args.db) as catalog:
        print(json.dumps(catalog.stats(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve
    serve(host=args.host, port=args.port, db_path=args.db, use_llm=not args.no_llm)
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sneakerrag",
        description="RAG agent that finds the cheapest listing for a Nike / adidas / "
                    "New Balance sneaker across six retail sites.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite catalogue path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sources", help="list the retail sites the agent knows")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("ingest", help="pull listings into the catalogue")
    p.add_argument("--live", action="store_true",
                   help="fetch real pages instead of the bundled sample data")
    p.add_argument("--sources", default="", help="comma-separated site keys")
    p.add_argument("--query", action="append", help="search term (repeatable, live mode)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("compare", help="price table for a shoe across sites")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("ask", help="ask a question in plain English")
    p.add_argument("question", nargs="+")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--no-llm", action="store_true", help="skip Claude, use the template writer")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("product", help="all listings for one style code")
    p.add_argument("style_code")
    p.set_defaults(func=cmd_product)

    p = sub.add_parser("deals", help="biggest discounts in the catalogue")
    p.add_argument("--min-pct", type=float, default=10.0)
    p.add_argument("--brand", default="")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_deals)

    p = sub.add_parser("stats", help="catalogue summary")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="run the local JSON API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-llm", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.db = Path(args.db)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

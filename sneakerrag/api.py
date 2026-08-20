"""Minimal JSON API over the agent (stdlib only, no web framework).

Endpoints::

    GET /health
    GET /sources
    GET /stats
    GET /compare?q=air+max+90&size=10.5&limit=3
    GET /ask?q=cheapest+air+max+90+in+size+10.5
    GET /product?style_code=DD1391-100
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .agent import SneakerAgent
from .matching import cluster_listings
from .models import fmt_money
from .normalize import normalize_style_code
from .sources import SITES
from .store import DEFAULT_DB


def make_handler(agent: SneakerAgent):
    class Handler(BaseHTTPRequestHandler):
        server_version = "sneakerrag/0.1"

        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            # Route through logging so importing the server (e.g. in tests)
            # doesn't spray the console.
            logging.getLogger("sneakerrag.api").info("%s %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:                             # noqa: N802
            url = urlsplit(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path.rstrip("/") or "/"
            try:
                if path == "/health":
                    return self._send({"ok": True, "listings": len(agent.index),
                                       "llm": getattr(agent.llm, "name", "none")})
                if path == "/sources":
                    return self._send({"sources": [
                        {"key": s.key, "name": s.name, "home": s.home, "kind": s.kind,
                         "brands": list(s.brands) or ["nike", "adidas", "new balance"],
                         "notes": s.notes}
                        for s in SITES]})
                if path == "/stats":
                    return self._send(agent.catalog.stats())
                if path in ("/compare", "/ask"):
                    query = params.get("q", "").strip()
                    if not query:
                        return self._send({"error": "missing q parameter"}, 400)
                    limit = int(params.get("limit", 3))
                    if path == "/ask":
                        return self._send(agent.answer(query, limit=limit).to_dict())
                    # Resolve size/condition before comparing so retrieval and
                    # ranking both see the constraint.
                    spec = agent.understand(query)
                    if params.get("size"):
                        spec.size = params["size"]
                    if params.get("condition"):
                        spec.condition = params["condition"]
                    spec, products = agent.compare(query, spec=spec, limit=limit)
                    return self._send({
                        "query": query, "spec": spec.to_dict(),
                        "products": [{
                            "product": p.display_name, "style_code": p.style_code,
                            "at_retail": p.at_retail,
                            "cheapest": _offer_json(p.best_offer(size=spec.size,
                                                                condition=spec.condition),
                                                    spec.size),
                            "offers": [_offer_json(o, spec.size)
                                       for o in p.offers(size=spec.size,
                                                         condition=spec.condition)],
                        } for p in products]})
                if path == "/product":
                    code = normalize_style_code(params.get("style_code", ""))
                    listings = [l for l in agent.catalog.listings()
                                if normalize_style_code(l.style_code) == code]
                    if not listings:
                        return self._send({"error": f"no listings for {code}"}, 404)
                    product = cluster_listings(listings)[0]
                    return self._send({
                        "product": product.display_name, "style_code": product.style_code,
                        "at_retail": product.at_retail,
                        "offers": [_offer_json(o) for o in product.offers(in_stock_only=False)]})
                self._send({"error": "not found",
                            "endpoints": ["/health", "/sources", "/stats", "/compare",
                                          "/ask", "/product"]}, 404)
            except Exception as exc:                          # keep the server alive
                self._send({"error": str(exc)}, 500)

    return Handler


def _offer_json(listing, size: str = "") -> dict | None:
    if listing is None:
        return None
    return {
        "source": listing.source_name, "source_kind": listing.source_kind,
        "url": listing.url, "title": listing.title,
        "price": listing.price_for(size), "from_price": listing.price,
        "shipping": listing.shipping, "fees": listing.fees,
        "total_price": listing.total_for(size), "currency": listing.currency,
        "display_price": fmt_money(listing.total_for(size), listing.currency),
        "size": size or None, "size_prices": listing.size_prices or None,
        "condition": listing.condition, "in_stock": listing.in_stock,
        "sizes": listing.sizes, "list_price": listing.list_price,
        "discount_pct": listing.discount_pct_for(size),
        "premium_pct": listing.premium_pct_for(size),
        "style_code": listing.style_code,
    }


def build_server(host: str = "127.0.0.1", port: int = 8000,
                 db_path: Path | str = DEFAULT_DB, use_llm: bool = True,
                 agent: SneakerAgent | None = None) -> tuple[ThreadingHTTPServer, SneakerAgent]:
    """Create (but do not start) the API server. Port 0 picks a free port."""
    agent = agent or SneakerAgent(db_path=db_path, use_llm=use_llm)
    return ThreadingHTTPServer((host, port), make_handler(agent)), agent


def serve(host: str = "127.0.0.1", port: int = 8000,
          db_path: Path | str = DEFAULT_DB, use_llm: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server, agent = build_server(host, port, db_path, use_llm)
    print(f"sneakerrag API on http://{host}:{port}  "
          f"({len(agent.index)} listings, llm={getattr(agent.llm, 'name', 'none')})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()

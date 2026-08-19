"""SQLite-backed catalogue: listings, cached embeddings and price history.

Price history is a first-class table rather than an afterthought — "is this
actually a good price?" is unanswerable from a single snapshot, and re-running
``ingest`` on a schedule turns the same store into a price tracker.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import Listing, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id   TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    brand        TEXT,
    model        TEXT,
    colorway     TEXT,
    style_code   TEXT,
    retailer_sku TEXT,
    price        REAL NOT NULL,
    list_price   REAL,
    currency     TEXT,
    shipping     REAL,
    in_stock     INTEGER,
    sizes        TEXT,
    gender       TEXT,
    condition    TEXT,
    image        TEXT,
    scraped_at   TEXT,
    raw          TEXT,
    vector       TEXT
);
CREATE INDEX IF NOT EXISTS idx_listings_brand ON listings(brand);
CREATE INDEX IF NOT EXISTS idx_listings_code  ON listings(style_code);
CREATE INDEX IF NOT EXISTS idx_listings_src   ON listings(source);

CREATE TABLE IF NOT EXISTS price_history (
    listing_id  TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    price       REAL NOT NULL,
    shipping    REAL,
    in_stock    INTEGER,
    PRIMARY KEY (listing_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_hist_listing ON price_history(listing_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

DEFAULT_DB = Path("data/catalog.db")
_COLUMNS = [
    "listing_id", "source", "source_name", "url", "title", "brand", "model",
    "colorway", "style_code", "retailer_sku", "price", "list_price", "currency",
    "shipping", "in_stock", "sizes", "gender", "condition", "image",
    "scraped_at", "raw", "vector",
]


class Catalog:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the threaded HTTP API can share one
        # catalogue; every statement below runs under self._lock.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- writes -----------------------------------------------------------
    def upsert_listings(self, listings: Iterable[Listing],
                        vectors: Sequence[Sequence[float]] | None = None) -> dict[str, int]:
        """Insert or update listings and append a price-history point.

        Returns counts of new / updated / price-changed rows.
        """
        items = list(listings)
        vecs = list(vectors) if vectors is not None else [None] * len(items)
        stats = {"new": 0, "updated": 0, "price_changed": 0}
        now = utcnow()
        with self._lock, closing(self.conn.cursor()) as cur:
            for listing, vector in zip(items, vecs):
                cur.execute("SELECT price FROM listings WHERE listing_id = ?", (listing.listing_id,))
                row = cur.fetchone()
                if row is None:
                    stats["new"] += 1
                else:
                    stats["updated"] += 1
                    if abs(float(row["price"]) - listing.price) > 0.005:
                        stats["price_changed"] += 1

                values = (
                    listing.listing_id, listing.source, listing.source_name, listing.url,
                    listing.title, listing.brand, listing.model, listing.colorway,
                    listing.style_code, listing.retailer_sku, listing.price, listing.list_price,
                    listing.currency, listing.shipping, int(listing.in_stock),
                    json.dumps(listing.sizes), listing.gender, listing.condition, listing.image,
                    listing.scraped_at, json.dumps(listing.raw, default=str),
                    json.dumps(list(vector)) if vector is not None else None,
                )
                placeholders = ", ".join("?" * len(_COLUMNS))
                cur.execute(
                    f"INSERT INTO listings ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(listing_id) DO UPDATE SET "
                    + ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "listing_id")
                    + (" , vector=COALESCE(excluded.vector, listings.vector)" if vector is None else ""),
                    values,
                )
                cur.execute(
                    "INSERT OR REPLACE INTO price_history "
                    "(listing_id, observed_at, price, shipping, in_stock) VALUES (?, ?, ?, ?, ?)",
                    (listing.listing_id, listing.scraped_at or now, listing.price,
                     listing.shipping, int(listing.in_stock)),
                )
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_ingest', ?)", (now,))
        self.conn.commit()
        return stats

    # -- reads ------------------------------------------------------------
    def _row_to_listing(self, row: sqlite3.Row) -> Listing:
        return Listing(
            listing_id=row["listing_id"], source=row["source"], source_name=row["source_name"],
            url=row["url"], title=row["title"], brand=row["brand"] or "", model=row["model"] or "",
            colorway=row["colorway"] or "", style_code=row["style_code"] or "",
            retailer_sku=row["retailer_sku"] or "", price=row["price"], list_price=row["list_price"],
            currency=row["currency"] or "USD", shipping=row["shipping"],
            in_stock=bool(row["in_stock"]), sizes=json.loads(row["sizes"] or "[]"),
            gender=row["gender"] or "", condition=row["condition"] or "new", image=row["image"] or "",
            scraped_at=row["scraped_at"] or "", raw=json.loads(row["raw"] or "{}"),
        )

    def listings(self, brand: str = "", source: str = "") -> list[Listing]:
        sql = "SELECT * FROM listings"
        clauses, params = [], []
        if brand:
            clauses.append("brand = ?")
            params.append(brand)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock, closing(self.conn.cursor()) as cur:
            return [self._row_to_listing(r) for r in cur.execute(sql, params)]

    def listings_with_vectors(self) -> tuple[list[Listing], list[list[float] | None]]:
        with self._lock, closing(self.conn.cursor()) as cur:
            rows = list(cur.execute("SELECT * FROM listings"))
        listings = [self._row_to_listing(r) for r in rows]
        vectors = [json.loads(r["vector"]) if r["vector"] else None for r in rows]
        return listings, vectors

    def price_history(self, listing_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(
                "SELECT observed_at, price, shipping, in_stock FROM price_history "
                "WHERE listing_id = ? ORDER BY observed_at", (listing_id,)).fetchall()
        return [dict(r) for r in rows]

    def price_drops(self, min_pct: float = 5.0) -> list[dict[str, Any]]:
        """Listings whose latest price is below their observed maximum."""
        out = []
        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(
                "SELECT listing_id, MAX(price) AS high, MIN(price) AS low, COUNT(*) AS n "
                "FROM price_history GROUP BY listing_id HAVING n > 1").fetchall()
            for row in rows:
                latest = cur.execute(
                    "SELECT price FROM price_history WHERE listing_id = ? "
                    "ORDER BY observed_at DESC LIMIT 1", (row["listing_id"],)).fetchone()
                if not latest or not row["high"]:
                    continue
                pct = (1 - latest["price"] / row["high"]) * 100
                if pct >= min_pct:
                    out.append({"listing_id": row["listing_id"], "high": row["high"],
                                "now": latest["price"], "drop_pct": round(pct, 1)})
        out.sort(key=lambda d: -d["drop_pct"])
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock, closing(self.conn.cursor()) as cur:
            total = cur.execute("SELECT COUNT(*) c FROM listings").fetchone()["c"]
            by_source = {r["source"]: r["c"] for r in cur.execute(
                "SELECT source, COUNT(*) c FROM listings GROUP BY source ORDER BY c DESC")}
            by_brand = {r["brand"] or "?": r["c"] for r in cur.execute(
                "SELECT brand, COUNT(*) c FROM listings GROUP BY brand ORDER BY c DESC")}
            points = cur.execute("SELECT COUNT(*) c FROM price_history").fetchone()["c"]
            last = cur.execute("SELECT value FROM meta WHERE key='last_ingest'").fetchone()
        return {"listings": total, "by_source": by_source, "by_brand": by_brand,
                "price_points": points, "last_ingest": last["value"] if last else None,
                "db": str(self.path)}

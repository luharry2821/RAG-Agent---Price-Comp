import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sneakerrag.store import Catalog
from tests.helpers import air_max_90_across_sites


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.catalog = Catalog(Path(self.tmp.name) / "test.db")
        self.listings = air_max_90_across_sites()

    def tearDown(self):
        self.catalog.close()
        self.tmp.cleanup()

    def test_insert_then_update_is_idempotent(self):
        first = self.catalog.upsert_listings(self.listings)
        second = self.catalog.upsert_listings(self.listings)
        self.assertEqual(first["new"], 4)
        self.assertEqual(second["new"], 0)
        self.assertEqual(second["updated"], 4)
        self.assertEqual(len(self.catalog.listings()), 4)

    def test_round_trip_preserves_fields(self):
        self.catalog.upsert_listings(self.listings)
        stored = {l.listing_id: l for l in self.catalog.listings()}
        for original in self.listings:
            got = stored[original.listing_id]
            self.assertEqual(got.price, original.price)
            self.assertEqual(got.sizes, original.sizes)
            self.assertEqual(got.style_code, original.style_code)
            self.assertEqual(got.in_stock, original.in_stock)

    def test_price_history_and_drops(self):
        self.catalog.upsert_listings(self.listings)
        cheaper = [replace(l, price=round(l.price * 0.8, 2), scraped_at="2099-01-01T12:00:00+00:00")
                   for l in self.listings]
        stats = self.catalog.upsert_listings(cheaper)
        self.assertEqual(stats["price_changed"], 4)
        history = self.catalog.price_history(self.listings[0].listing_id)
        self.assertEqual(len(history), 2)
        drops = self.catalog.price_drops(min_pct=10)
        self.assertEqual(len(drops), 4)
        self.assertAlmostEqual(drops[0]["drop_pct"], 20.0, places=0)

    def test_resale_fields_survive_the_round_trip(self):
        self.catalog.upsert_listings(self.listings)
        stored = {l.source: l for l in self.catalog.listings()}
        goat = stored["goat"]
        self.assertEqual(goat.source_kind, "resale")
        self.assertEqual(goat.fees, 5.0)
        self.assertEqual(goat.size_prices["10.5"], 118.0)
        self.assertEqual(goat.price_for("10.5"), 118.0)
        self.assertEqual(stored["nike"].source_kind, "brand")

    def test_vectors_cached_and_reused(self):
        vectors = [[0.5] * 8 for _ in self.listings]
        self.catalog.upsert_listings(self.listings, vectors)
        _, stored = self.catalog.listings_with_vectors()
        self.assertTrue(all(v == [0.5] * 8 for v in stored))

    def test_filters_and_stats(self):
        self.catalog.upsert_listings(self.listings)
        self.assertEqual(len(self.catalog.listings(brand="nike")), 4)
        self.assertEqual(len(self.catalog.listings(source="stadiumgoods")), 1)
        stats = self.catalog.stats()
        self.assertEqual(stats["listings"], 4)
        self.assertEqual(stats["by_brand"]["nike"], 4)


if __name__ == "__main__":
    unittest.main()

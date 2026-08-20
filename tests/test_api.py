import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from sneakerrag.agent import SneakerAgent
from sneakerrag.api import build_server
from sneakerrag.sources import get_sources
from sneakerrag.store import Catalog


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog = Catalog(Path(cls.tmp.name) / "api.db")
        agent = SneakerAgent(catalog=cls.catalog, use_llm=False)
        agent.ingest(get_sources())
        cls.server, _ = build_server("127.0.0.1", 0, agent=agent)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.catalog.close()
        cls.tmp.cleanup()

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as fh:
            return fh.status, json.loads(fh.read())

    def test_health(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertGreater(body["listings"], 0)

    def test_sources_lists_every_site(self):
        _, body = self.get("/sources")
        keys = {s["key"] for s in body["sources"]}
        self.assertEqual(keys, {"nike", "adidas", "newbalance", "stadiumgoods",
                                "flightclub", "goat", "kickscrew"})
        self.assertEqual({s["kind"] for s in body["sources"]}, {"brand", "resale"})

    def test_compare_prices_the_requested_size(self):
        _, body = self.get("/compare?q=dunk+low&size=10&limit=1")
        offers = body["products"][0]["offers"]
        self.assertTrue(offers)
        for offer in offers:
            self.assertEqual(offer["size"], "10")
            if offer["size_prices"]:
                self.assertEqual(offer["price"], offer["size_prices"]["10"])

    def test_compare_returns_a_cheapest_offer(self):
        _, body = self.get("/compare?q=air+max+90&limit=1")
        product = body["products"][0]
        cheapest = product["cheapest"]
        self.assertIsNotNone(cheapest)
        self.assertEqual(cheapest["total_price"],
                         min(o["total_price"] for o in product["offers"]))

    def test_ask_returns_answer_with_citations(self):
        _, body = self.get("/ask?q=cheapest+samba+og&limit=1")
        self.assertTrue(body["answer"])
        self.assertTrue(body["citations"])

    def test_product_lookup_and_404(self):
        _, body = self.get("/product?style_code=B75806")
        self.assertTrue(body["offers"])
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/product?style_code=ZZ0000-000")
        self.assertEqual(ctx.exception.code, 404)

    def test_missing_query_is_a_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/compare")
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()

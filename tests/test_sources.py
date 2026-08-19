import json
import tempfile
import unittest
from pathlib import Path

from sneakerrag.sources import SITES, SITES_BY_KEY, get_sources, load_fixture
from sneakerrag.sources.base import LiveSource, build_listing
from sneakerrag.sources.jsonld import extract_listing_fields, find_products

PRODUCT_HTML = """
<html><head>
<meta property="og:image" content="https://img.example/fallback.jpg">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product",
 "name":"Nike Air Max 90 Men's Shoes","brand":{"@type":"Brand","name":"Nike"},
 "mpn":"HM0089-100","sku":"FL-99213","color":"White/Black",
 "image":"https://img.example/am90.jpg",
 "offers":[{"@type":"Offer","price":"119.99","priceCurrency":"USD",
            "availability":"https://schema.org/InStock","url":"https://www.footlocker.com/product/~x/HM0089-100.html"},
           {"@type":"Offer","price":"129.99","priceCurrency":"USD","availability":"InStock"}]}
</script></head><body></body></html>
"""

SEARCH_HTML = """
<html><body>
<a href="/product/~nike-air-max-90/HM0089100.html">Air Max 90</a>
<a href="/help/returns">Returns</a>
<a href="/product/~nike-air-max-90/HM0089100.html#reviews">reviews anchor</a>
</body></html>
"""


class FakeClient:
    """Stands in for HttpClient so the live path is testable offline."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def get(self, url: str, **kw) -> str:
        self.requested.append(url)
        for key, body in self.pages.items():
            if key in url:
                return body
        raise AssertionError(f"unexpected fetch: {url}")


class TestJsonLd(unittest.TestCase):
    def test_extracts_product_and_cheapest_offer(self):
        data = extract_listing_fields(PRODUCT_HTML)
        self.assertEqual(data["title"], "Nike Air Max 90 Men's Shoes")
        self.assertEqual(data["style_code"], "HM0089-100")
        self.assertEqual(data["price"], "119.99")       # cheapest of the two offers
        self.assertTrue(data["in_stock"])

    def test_meta_fallback_when_no_jsonld(self):
        html = ('<meta property="og:title" content="adidas Samba OG">'
                '<meta property="product:price:amount" content="99.99">')
        data = extract_listing_fields(html)
        self.assertEqual(data["title"], "adidas Samba OG")
        self.assertEqual(data["price"], "99.99")

    def test_malformed_json_does_not_raise(self):
        self.assertEqual(find_products('<script type="application/ld+json">{oops</script>'), [])


class TestBuildListing(unittest.TestCase):
    spec = SITES_BY_KEY["footlocker"]

    def test_normalizes_into_a_listing(self):
        listing = build_listing(self.spec, extract_listing_fields(PRODUCT_HTML))
        self.assertEqual(listing.brand, "nike")
        self.assertEqual(listing.model, "air max 90")
        self.assertEqual(listing.style_code, "HM0089-100")
        self.assertEqual(listing.price, 119.99)
        self.assertEqual(listing.source_name, "Foot Locker")

    def test_rejects_records_without_a_price_or_url(self):
        self.assertIsNone(build_listing(self.spec, {"title": "x", "url": "/p/1"}))
        self.assertIsNone(build_listing(self.spec, {"title": "", "url": "", "price": 10}))

    def test_single_brand_store_supplies_the_brand(self):
        listing = build_listing(SITES_BY_KEY["adidas"],
                                {"title": "Samba OG Shoes", "url": "/us/samba/B75806.html",
                                 "price": "100.00"})
        self.assertEqual(listing.brand, "adidas")

    def test_relative_urls_are_absolute(self):
        listing = build_listing(self.spec, {"title": "Nike Dunk Low", "url": "/product/x.html",
                                            "price": "95"})
        self.assertTrue(listing.url.startswith("https://www.footlocker.com/"))


class TestLiveSource(unittest.TestCase):
    def test_search_page_to_listings(self):
        client = FakeClient({"search?query=": SEARCH_HTML, "/product/": PRODUCT_HTML})
        source = LiveSource(SITES_BY_KEY["footlocker"], client=client)
        listings = source.search("air max 90", limit=5)
        self.assertEqual(len(listings), 1)              # anchor duplicate collapsed
        self.assertEqual(listings[0].style_code, "HM0089-100")
        self.assertEqual(len(client.requested), 2)      # one search + one product page


class TestFixtures(unittest.TestCase):
    def test_every_site_has_sample_data(self):
        for spec in SITES:
            source = load_fixture(spec)
            self.assertTrue(source.all(), f"no fixtures for {spec.key}")
            for listing in source.all():
                self.assertIn(listing.brand, ("nike", "adidas", "new balance"))
                self.assertTrue(spec.sells(listing.brand))
                self.assertGreater(listing.price, 0)

    def test_get_sources_defaults_to_fixtures(self):
        adapters = get_sources()
        self.assertEqual(len(adapters), len(SITES))
        self.assertGreater(sum(len(a.search("", limit=99)) for a in adapters), 20)

    def test_unknown_source_key_raises(self):
        with self.assertRaises(KeyError):
            get_sources(["nordstrom"])

    def test_missing_fixture_file_is_empty_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = load_fixture(SITES_BY_KEY["nike"], Path(tmp))
            self.assertEqual(source.all(), [])

    def test_fixture_files_are_marked_synthetic(self):
        for spec in SITES:
            payload = json.loads(Path(f"data/fixtures/{spec.key}.json").read_text())
            self.assertIn("SYNTHETIC", payload["_note"])


if __name__ == "__main__":
    unittest.main()

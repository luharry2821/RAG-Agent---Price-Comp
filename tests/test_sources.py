import json
import tempfile
import unittest
from pathlib import Path

from sneakerrag.sources import SITES, SITES_BY_KEY, get_sources, load_fixture
from sneakerrag.sources.base import LiveSource, build_listing
from sneakerrag.sources.jsonld import extract_listing_fields, find_products

RETAIL_HTML = """
<html><head>
<meta property="og:image" content="https://img.example/fallback.jpg">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product",
 "name":"Nike Air Max 90 Men's Shoes","brand":{"@type":"Brand","name":"Nike"},
 "mpn":"HM0089-100","sku":"NK-99213","color":"White/Black",
 "image":"https://img.example/am90.jpg",
 "offers":[{"@type":"Offer","price":"119.99","priceCurrency":"USD",
            "availability":"https://schema.org/InStock","url":"https://www.nike.com/t/x/HM0089-100"},
           {"@type":"Offer","price":"129.99","priceCurrency":"USD","availability":"InStock"}]}
</script></head><body></body></html>
"""

# How a resale marketplace publishes it: one offer per size, under AggregateOffer.
RESALE_HTML = """
<html><body>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product",
 "name":"Nike Dunk Low Retro 'White/Black'","brand":"Nike","mpn":"DD1391-100",
 "description":"Used - good condition",
 "offers":{"@type":"AggregateOffer","lowPrice":"148","priceCurrency":"USD","offers":[
   {"@type":"Offer","name":"Size 9","price":"152","availability":"InStock",
    "url":"https://www.goat.com/sneakers/nike-dunk-low-retro-dd1391-100"},
   {"@type":"Offer","name":"Size 10.5","price":"148","availability":"InStock"},
   {"@type":"Offer","name":"Size 13","price":"215","availability":"InStock"}]}}
</script></body></html>
"""

SEARCH_HTML = """
<html><body>
<a href="/sneakers/nike-dunk-low-retro-dd1391-100">Dunk Low</a>
<a href="/help/returns">Returns</a>
<a href="/sneakers/nike-dunk-low-retro-dd1391-100#reviews">reviews anchor</a>
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


class TestSiteSet(unittest.TestCase):
    def test_three_brand_stores_and_four_resale_marketplaces(self):
        self.assertEqual([s.key for s in SITES if s.kind == "brand"],
                         ["nike", "adidas", "newbalance"])
        self.assertEqual([s.key for s in SITES if s.kind == "resale"],
                         ["stadiumgoods", "flightclub", "goat", "kickscrew"])

    def test_brand_stores_only_carry_their_own_brand(self):
        self.assertTrue(SITES_BY_KEY["nike"].sells("nike"))
        self.assertFalse(SITES_BY_KEY["nike"].sells("adidas"))
        self.assertTrue(SITES_BY_KEY["goat"].sells("adidas"))


class TestJsonLd(unittest.TestCase):
    def test_extracts_product_and_cheapest_offer(self):
        data = extract_listing_fields(RETAIL_HTML)
        self.assertEqual(data["title"], "Nike Air Max 90 Men's Shoes")
        self.assertEqual(data["style_code"], "HM0089-100")
        self.assertEqual(data["price"], "119.99")       # cheapest of the two offers
        self.assertTrue(data["in_stock"])

    def test_per_size_offers_become_a_price_curve(self):
        data = extract_listing_fields(RESALE_HTML)
        self.assertEqual(data["size_prices"], {"9": 152.0, "10.5": 148.0, "13": 215.0})
        self.assertEqual(data["condition"], "used")

    def test_meta_fallback_when_no_jsonld(self):
        html = ('<meta property="og:title" content="adidas Samba OG">'
                '<meta property="product:price:amount" content="99.99">')
        data = extract_listing_fields(html)
        self.assertEqual(data["title"], "adidas Samba OG")
        self.assertEqual(data["price"], "99.99")

    def test_malformed_json_does_not_raise(self):
        self.assertEqual(find_products('<script type="application/ld+json">{oops</script>'), [])


class TestBuildListing(unittest.TestCase):
    spec = SITES_BY_KEY["goat"]

    def test_normalizes_into_a_listing(self):
        listing = build_listing(SITES_BY_KEY["nike"], extract_listing_fields(RETAIL_HTML))
        self.assertEqual(listing.brand, "nike")
        self.assertEqual(listing.model, "air max 90")
        self.assertEqual(listing.style_code, "HM0089-100")
        self.assertEqual(listing.price, 119.99)
        self.assertEqual(listing.source_kind, "brand")

    def test_resale_listing_keeps_the_size_curve(self):
        data = extract_listing_fields(RESALE_HTML)
        data.update(shipping="12.00", fees="5.00")
        listing = build_listing(self.spec, data)
        self.assertEqual(listing.source_kind, "resale")
        self.assertEqual(listing.price, 148.0)              # headline = lowest ask
        self.assertEqual(listing.price_for("13"), 215.0)
        self.assertEqual(listing.total_for("13"), 232.0)    # + shipping + fees
        self.assertEqual(listing.condition, "used")
        self.assertEqual(sorted(listing.sizes, key=float), ["9", "10.5", "13"])

    def test_rejects_records_without_a_price_or_url(self):
        self.assertIsNone(build_listing(self.spec, {"title": "x", "url": "/p/1"}))
        self.assertIsNone(build_listing(self.spec, {"title": "", "url": "", "price": 10}))

    def test_single_brand_store_supplies_the_brand(self):
        listing = build_listing(SITES_BY_KEY["adidas"],
                                {"title": "Samba OG Shoes", "url": "/us/samba/B75806.html",
                                 "price": "100.00"})
        self.assertEqual(listing.brand, "adidas")

    def test_relative_urls_are_absolute(self):
        listing = build_listing(self.spec, {"title": "Nike Dunk Low", "url": "/sneakers/x",
                                            "price": "95"})
        self.assertTrue(listing.url.startswith("https://www.goat.com/"))


class TestLiveSource(unittest.TestCase):
    def test_search_page_to_listings(self):
        client = FakeClient({"search?query=": SEARCH_HTML, "/sneakers/": RESALE_HTML})
        source = LiveSource(SITES_BY_KEY["goat"], client=client)
        listings = source.search("dunk low", limit=5)
        self.assertEqual(len(listings), 1)              # anchor duplicate collapsed
        self.assertEqual(listings[0].style_code, "DD1391-100")
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

    def test_resale_fixtures_price_per_size(self):
        for spec in SITES:
            for listing in load_fixture(spec).all():
                if spec.kind == "resale":
                    self.assertTrue(listing.size_prices, f"{spec.key} has no size curve")
                    self.assertEqual(listing.price, min(listing.size_prices.values()))
                else:
                    self.assertEqual(listing.size_prices, {})

    def test_get_sources_defaults_to_fixtures(self):
        adapters = get_sources()
        self.assertEqual(len(adapters), len(SITES))
        self.assertGreater(sum(len(a.search("", limit=99)) for a in adapters), 20)

    def test_unknown_source_key_raises(self):
        with self.assertRaises(KeyError):
            get_sources(["footlocker"])

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

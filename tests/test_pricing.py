"""Resale-marketplace economics: per-size asks, fees, condition, MSRP premium."""

import tempfile
import unittest
from pathlib import Path

from sneakerrag.agent import SneakerAgent
from sneakerrag.matching import cluster_listings
from sneakerrag.store import Catalog
from tests.helpers import air_max_90_across_sites, listing


class TestListingPricing(unittest.TestCase):
    def setUp(self):
        self.goat = listing("goat", "Nike Dunk Low Retro 'White/Black'", 148.0,
                            style_code="DD1391-100", shipping=12.0, fees=5.0,
                            list_price=120.0,
                            size_prices={"9": 152.0, "10.5": 148.0, "13": 215.0})

    def test_price_is_looked_up_per_size(self):
        self.assertEqual(self.goat.price_for("10.5"), 148.0)
        self.assertEqual(self.goat.price_for("13"), 215.0)

    def test_unknown_size_falls_back_to_the_headline_ask(self):
        self.assertEqual(self.goat.price_for("14"), 148.0)
        self.assertEqual(self.goat.price_for(""), 148.0)

    def test_delivered_price_includes_shipping_and_fees(self):
        self.assertEqual(self.goat.extras, 17.0)
        self.assertEqual(self.goat.total_for("13"), 232.0)
        self.assertEqual(self.goat.total_price, 165.0)      # at the lowest ask

    def test_premium_over_msrp_is_reported_per_size(self):
        self.assertEqual(self.goat.premium_pct_for("13"), 79.2)
        self.assertEqual(self.goat.discount_pct_for("13"), 0.0)

    def test_size_availability_covers_the_price_curve(self):
        self.assertTrue(self.goat.has_size("13"))
        self.assertFalse(self.goat.has_size("7"))

    def test_retail_listing_is_flat_priced(self):
        nike = listing("nike", "Nike Dunk Low Retro Men's Shoes", 120.0)
        self.assertEqual(nike.price_for("13"), 120.0)
        self.assertFalse(nike.is_resale)


class TestSizeAwareComparison(unittest.TestCase):
    def setUp(self):
        # Same shoe, three sellers, crossing price curves: KicksCrew wins a 9,
        # GOAT wins a 13, and the brand store wins neither.
        self.product = cluster_listings([
            listing("nike", "Nike Dunk Low Retro Men's Shoes (DD1391-100)", 120.0,
                    sizes=["9"], in_stock=False),
            listing("goat", "Nike Dunk Low Retro 'White/Black' (DD1391-100)", 150.0,
                    shipping=12.0, fees=5.0, list_price=120.0,
                    size_prices={"9": 180.0, "13": 150.0}),
            listing("kickscrew", "Nike Dunk Low Retro White/Black DD1391-100", 155.0,
                    shipping=0.0, list_price=120.0,
                    size_prices={"9": 155.0, "13": 205.0}),
        ])[0]

    def test_cheapest_seller_changes_with_the_size(self):
        self.assertEqual(self.product.best_offer(size="9").source, "kickscrew")
        self.assertEqual(self.product.best_offer(size="13").source, "goat")

    def test_savings_are_computed_for_that_size(self):
        self.assertEqual(self.product.savings(size="9"), 42.0)     # 197 - 155
        self.assertEqual(self.product.savings(size="13"), 38.0)    # 205 - 167

    def test_price_range_is_size_specific(self):
        self.assertEqual(self.product.price_range("13"), (167.0, 205.0))

    def test_sold_out_at_retail_is_visible(self):
        self.assertFalse(self.product.at_retail)
        self.assertTrue(self.product.has_resale_listings)

    def test_fees_and_shipping_flip_the_winner(self):
        # GOAT asks $150 for a 13 but adds $17 of shipping and fees; Stadium
        # Goods asks $160 and ships free. The ask says GOAT, the receipt says
        # Stadium Goods.
        rival = listing("stadiumgoods", "Nike Dunk Low Retro 'White/Black'", 160.0,
                        style_code="DD1391-100", shipping=0.0, size_prices={"13": 160.0})
        product = cluster_listings(list(self.product.listings) + [rival])[0]
        self.assertEqual(product.best_offer(size="13").source, "stadiumgoods")   # 160 delivered
        self.assertEqual(product.best_offer(size="13", include_shipping=False).source,
                         "goat")                                                 # 150 ask


class TestAgentResaleBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog = Catalog(Path(cls.tmp.name) / "resale.db")
        cls.catalog.upsert_listings(air_max_90_across_sites() + [
            listing("flightclub", "Nike Air Max 90 White/Black", 95.0, style_code="HM0089-100",
                    shipping=14.5, condition="used", list_price=130.0,
                    size_prices={"9": 95.0, "10": 99.0, "10.5": 104.0, "11": 97.0}),
        ])
        cls.agent = SneakerAgent(catalog=cls.catalog, use_llm=False)

    @classmethod
    def tearDownClass(cls):
        cls.catalog.close()
        cls.tmp.cleanup()

    def test_used_listing_wins_on_price_but_is_labelled(self):
        answer = self.agent.answer("cheapest air max 90 in a 10.5", limit=1)
        self.assertIn("Flight Club", answer.text)
        self.assertIn("used", answer.text)

    def test_new_only_request_excludes_the_used_listing(self):
        spec = self.agent.understand("cheapest brand new air max 90 in a 10.5")
        self.assertEqual(spec.condition, "new")
        _, products = self.agent.compare("cheapest brand new air max 90 in a 10.5",
                                         spec=spec, limit=1)
        offers = products[0].offers(size="10.5", condition="new")
        self.assertTrue(offers)
        self.assertTrue(all(o.condition == "new" for o in offers))

    def test_answer_quotes_the_requested_size_not_the_lowest_ask(self):
        answer = self.agent.answer("cheapest air max 90 in a 10.5", limit=1)
        self.assertIn("for a US 10.5", answer.text)

    def test_budget_is_applied_to_the_requested_size(self):
        answer = self.agent.answer("air max 90 size 10.5 under $60", limit=1)
        self.assertIn("Nothing", answer.text)

    def test_different_sizes_produce_different_winners_end_to_end(self):
        _, small = self.agent.compare("air max 90", spec=self._spec("9"), limit=1)
        _, large = self.agent.compare("air max 90", spec=self._spec("11"), limit=1)
        self.assertEqual(small[0].best_offer(size="9").source, "flightclub")
        self.assertTrue(large[0].best_offer(size="11"))

    def _spec(self, size: str):
        spec = self.agent.understand("air max 90")
        spec.size = size
        return spec


if __name__ == "__main__":
    unittest.main()

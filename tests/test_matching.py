import unittest

from sneakerrag.matching import cluster_listings, is_same_product, match_score, model_similarity
from tests.helpers import air_max_90_across_sites, listing


class TestPairwise(unittest.TestCase):
    def test_style_code_is_authoritative(self):
        a = listing("nike", "Nike Air Max 90 Men's Shoes", 130, style_code="HM0089-100")
        b = listing("jdsports", "AM90 White Black", 99, style_code="HM0089-100")
        score, reason = match_score(a, b)
        self.assertEqual(score, 1.0)
        self.assertIn("HM0089-100", reason)

    def test_conflicting_style_codes_never_merge(self):
        a = listing("nike", "Nike Air Max 90 Men's Shoes", 130, style_code="HM0089-100")
        b = listing("footlocker", "Nike Air Max 90 - Men's", 120, style_code="CN8490-002")
        self.assertFalse(is_same_product(a, b))

    def test_version_numbers_block_a_fuzzy_merge(self):
        self.assertEqual(model_similarity("990v6", "990v5"), 0.0)
        a = listing("newbalance", "New Balance 990v6 Men's Grey", 200)
        b = listing("dicks", "New Balance Men's 990v5 Grey", 175)
        self.assertFalse(is_same_product(a, b))

    def test_qualifier_words_still_merge(self):
        a = listing("newbalance", "New Balance 990v6 Made in USA Men's Grey", 200)
        b = listing("dicks", "New Balance Men's 990v6 Grey", 190)
        self.assertTrue(is_same_product(a, b))

    def test_different_brands_never_merge(self):
        a = listing("footlocker", "Nike Air Max 90 - Men's - White/Black", 120)
        b = listing("footlocker", "adidas Air Max 90 - Men's - White/Black", 120)
        self.assertFalse(is_same_product(a, b))

    def test_mens_and_womens_are_separate_products(self):
        a = listing("footlocker", "Nike Pegasus 41 - Men's - Black/White", 140)
        b = listing("footlocker", "Nike Pegasus 41 - Women's - Black/White", 140)
        self.assertFalse(is_same_product(a, b))

    def test_different_colorways_do_not_merge(self):
        a = listing("footlocker", "Nike Dunk Low Retro - Men's - White/Black", 120)
        b = listing("footlocker", "Nike Dunk Low Retro - Men's - Green/Cream", 120)
        self.assertFalse(is_same_product(a, b))


class TestClustering(unittest.TestCase):
    def test_one_shoe_four_sites_one_cluster(self):
        products = cluster_listings(air_max_90_across_sites())
        self.assertEqual(len(products), 1)
        product = products[0]
        self.assertEqual(len(product.listings), 4)
        self.assertEqual(product.style_code, "HM0089-100")
        # JD is cheapest on sticker price but charges shipping, so the
        # delivered-price winner is Dick's.
        self.assertEqual(product.best_offer().source, "dickssportinggoods")
        self.assertEqual(product.best_offer(include_shipping=False).source, "jdsports")

    def test_distinct_shoes_stay_distinct(self):
        items = air_max_90_across_sites() + [
            listing("footlocker", "Nike Air Max 95 - Men's - Grey", 175),
            listing("footlocker", "Nike Air Max 90 - Men's - Infrared", 130),
        ]
        products = cluster_listings(items)
        # No cluster may contain two different colorways of the same silhouette:
        # quoting an infrared price for a white/black shoe is worse than not
        # grouping them at all.
        for product in products:
            colorways = {l.colorway for l in product.listings if l.colorway}
            self.assertLessEqual(len(colorways), 1, product.display_name)
        names = {p.display_name for p in products}
        self.assertIn("Nike Air Max 90 — Infrared", names)
        self.assertIn("Nike Air Max 95 — Grey", names)

    def test_cheapest_accounts_for_shipping(self):
        items = [
            listing("jdsports", "Nike Air Max 90 Men's", 109.99, shipping=25.0),
            listing("footlocker", "Nike Air Max 90 - Men's - White/Black", 119.99, shipping=0.0),
        ]
        product = cluster_listings(items)[0]
        self.assertEqual(product.best_offer().source, "footlocker")
        self.assertEqual(product.best_offer(include_shipping=False).source, "jdsports")


if __name__ == "__main__":
    unittest.main()

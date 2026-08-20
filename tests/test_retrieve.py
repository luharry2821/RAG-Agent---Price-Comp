import unittest

from sneakerrag.embeddings import HashingEmbedder, cosine
from sneakerrag.models import QuerySpec
from sneakerrag.retrieve import VectorIndex
from tests.helpers import air_max_90_across_sites, listing


def build_index() -> VectorIndex:
    index = VectorIndex()
    index.add(air_max_90_across_sites() + [
        listing("stadiumgoods", "adidas Samba OG - Cloud White/Core Black", 100,
                style_code="B75806"),
        listing("newbalance", "New Balance 990v6 Men's Grey", 200, style_code="M990GL6"),
        listing("goat", "Nike Dunk Low Retro Men's White/Black", 95,
                style_code="DD1391-100", in_stock=False),
    ])
    return index


class TestEmbedder(unittest.TestCase):
    def test_similar_titles_score_higher_than_unrelated(self):
        e = HashingEmbedder()
        near = cosine(e.embed("Nike Air Max 90 White Black"), e.embed("nike air max 90 mens"))
        far = cosine(e.embed("Nike Air Max 90 White Black"), e.embed("New Balance 990v6 grey"))
        self.assertGreater(near, 0.4)
        self.assertLess(far, near / 2)

    def test_vectors_are_unit_length(self):
        vec = HashingEmbedder().embed("adidas Samba OG")
        self.assertAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0, places=6)

    def test_empty_text_is_safe(self):
        self.assertEqual(cosine(HashingEmbedder().embed(""), HashingEmbedder().embed("x")), 0.0)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_finds_the_right_silhouette(self):
        hits = self.index.search("air max 90", QuerySpec(in_stock_only=False), top_k=4)
        self.assertTrue(all("air max 90" in h.listing.model for h in hits))

    def test_style_code_dominates_ranking(self):
        spec = QuerySpec(style_code="M990GL6", in_stock_only=False)
        hits = self.index.search("grey running shoe", spec, top_k=3)
        self.assertEqual(hits[0].listing.style_code, "M990GL6")

    def test_brand_filter_is_hard(self):
        hits = self.index.search("shoes", QuerySpec(brands=["adidas"], in_stock_only=False))
        self.assertEqual({h.listing.brand for h in hits}, {"adidas"})

    def test_stock_and_size_filters(self):
        self.assertTrue(all(h.listing.in_stock
                            for h in self.index.search("dunk low", QuerySpec(in_stock_only=True))))
        hits = self.index.search("air max 90", QuerySpec(size="10.5", in_stock_only=False))
        self.assertTrue(all(h.listing.has_size("10.5") for h in hits))

    def test_price_ceiling_uses_delivered_price(self):
        spec = QuerySpec(max_price=112.0, include_shipping=True, in_stock_only=False)
        hits = self.index.search("air max 90", spec)
        self.assertTrue(all(h.listing.total_price <= 112.0 for h in hits))

    def test_no_results_returns_empty_list(self):
        self.assertEqual(self.index.search("hiking boots", QuerySpec(brands=["puma"])), [])


if __name__ == "__main__":
    unittest.main()

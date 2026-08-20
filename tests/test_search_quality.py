"""Query understanding at the index level: vocabulary, typos, relevance floor."""

import unittest

from sneakerrag import retrieve as retrieve_module
from sneakerrag.aliases import (
    expand_query,
    meaningful_tokens,
    similarity,
    term_sources,
)
from sneakerrag.matching import cluster_listings
from sneakerrag.models import QuerySpec
from sneakerrag.retrieve import RERANK_DEPTH, VectorIndex
from sneakerrag.sources import get_sources


def fixture_index(**kwargs) -> VectorIndex:
    index = VectorIndex(**kwargs)
    index.add([l for adapter in get_sources() for l in adapter.search("", limit=999)])
    return index


def top_models(index: VectorIndex, query: str, k: int = 3) -> list[str]:
    hits = index.search(query, QuerySpec(in_stock_only=False), top_k=k)
    out: list[str] = []
    for hit in hits:
        if hit.listing.model not in out:
            out.append(hit.listing.model)
    return out


class TestVocabulary(unittest.TestCase):
    def test_nickname_expands_to_catalogue_words(self):
        terms = dict(expand_query("af1s"))
        self.assertIn("air", terms)
        self.assertIn("force", terms)
        self.assertLess(terms["air"], 1.0)          # aliases never outrank typed words

    def test_colour_slang(self):
        self.assertIn("white", dict(expand_query("panda dunks")))
        self.assertIn("black", dict(expand_query("panda dunks")))

    def test_plurals(self):
        self.assertIn("samba", dict(expand_query("sambas")))
        self.assertIn("dunk", dict(expand_query("dunks")))

    def test_known_misspellings(self):
        self.assertIn("adidas", dict(expand_query("adiddas sambas")))
        self.assertIn("balance", dict(expand_query("new balence 990")))

    def test_stopwords_are_dropped_from_content_words(self):
        self.assertEqual(meaningful_tokens("cheapest nike dunk low in a size 10"),
                         ["nike", "dunk", "low", "10"])

    def test_a_query_of_pure_noise_keeps_its_words(self):
        self.assertEqual(meaningful_tokens("cheapest shoes"), [])
        self.assertIn("shoes", dict(expand_query("cheapest shoes")))

    def test_trigram_similarity_catches_typos(self):
        self.assertGreater(similarity("ultrabost", "ultraboost"), 0.6)
        self.assertLess(similarity("samba", "ultraboost"), 0.2)


class TestRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = fixture_index()

    def test_nicknames_find_the_right_shoe(self):
        self.assertEqual(top_models(self.index, "af1s")[0], "air force 1 07")
        self.assertEqual(top_models(self.index, "panda dunks")[0], "dunk low retro")
        self.assertEqual(top_models(self.index, "sambas")[0], "samba og")
        self.assertEqual(top_models(self.index, "nb 990s")[0], "990v6")

    def test_typos_are_repaired_against_the_corpus(self):
        self.assertEqual(top_models(self.index, "ultrabost core black")[0], "ultraboost 5")
        self.assertEqual(top_models(self.index, "peagasus 41")[0], "pegasus 41")

    def test_noise_words_do_not_outvote_the_model_name(self):
        # "nike" and "size" match far more listings than "dunks" does.
        self.assertEqual(top_models(self.index, "cheapest nike dunks size 13")[0],
                         "dunk low retro")

    def test_style_code_query_wins_outright(self):
        hits = self.index.search("B75806", QuerySpec(style_code="B75806", in_stock_only=False))
        self.assertEqual(hits[0].listing.style_code, "B75806")
        self.assertEqual(hits[0].boost, 1.0)

    def test_field_weighting_prefers_the_model_over_the_title(self):
        hit = self.index.search("samba og", QuerySpec(in_stock_only=False))[0]
        self.assertEqual(hit.listing.model, "samba og")

    def test_matched_terms_are_reported(self):
        hit = self.index.search("990v6 grey", QuerySpec(in_stock_only=False))[0]
        self.assertIn("990v6", hit.terms)


class TestRelevanceFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = fixture_index()

    def test_shoes_we_do_not_carry_return_nothing(self):
        for query in ("asics gel kayano 31", "jordan 4 bred", "hoka clifton 9",
                      "converse chuck 70"):
            self.assertEqual(self.index.search(query, QuerySpec(in_stock_only=False)), [],
                             f"{query} should not match anything")

    def test_detailed_names_for_shoes_we_lack_return_nothing(self):
        for query in ("Air Jordan 1 Retro High OG 'Chicago Lost & Found",
                      "yeezy 350 v2 zebra",
                      "new balance 2002r protection pack"):
            self.assertEqual(self.index.search(query, QuerySpec(in_stock_only=False)), [],
                             f"{query} should not match anything")

    def test_an_expansion_cannot_vouch_for_itself(self):
        # "chicago" expands to "red white black". Those colours matching must
        # credit *chicago*, not add three independent matches.
        sources = term_sources("chicago")
        self.assertEqual(sources["white"], {"chicago"})
        self.assertEqual(sources["black"], {"chicago"})

    def test_matching_only_descriptive_words_is_not_enough(self):
        # Every catalogue has something black, low and retro.
        self.assertEqual(self.index.search("black low retro og premium",
                                           QuerySpec(in_stock_only=False)), [])

    def test_an_exact_style_code_outranks_the_floor(self):
        # A SKU is an identity: it should return the shoe even when the rest of
        # the words match nothing.
        hits = self.index.search("grey running shoe",
                                 QuerySpec(style_code="M990GL6", in_stock_only=False))
        self.assertEqual(hits[0].listing.style_code, "M990GL6")

    def test_a_browse_query_still_lists_the_catalogue(self):
        self.assertTrue(self.index.search("cheapest shoes", QuerySpec(in_stock_only=False)))

    def test_partial_matches_survive(self):
        # No Air Max 95 in stock, but the 90 is a legitimate near miss — the
        # agent labels it rather than the index hiding it.
        self.assertTrue(self.index.search("air max 95", QuerySpec(in_stock_only=False)))


class TestIndexMechanics(unittest.TestCase):
    def test_fusion_modes_all_rank_sensibly(self):
        for fusion in ("hybrid", "linear", "rrf"):
            index = fixture_index(fusion=fusion)
            self.assertEqual(top_models(index, "samba og")[0], "samba og", fusion)

    def test_incremental_adds_extend_the_index(self):
        adapters = get_sources()
        index = VectorIndex()
        first = adapters[0].search("", limit=999)
        index.add(first)
        self.assertEqual(len(index), len(first))
        rest = [l for a in adapters[1:] for l in a.search("", limit=999)]
        index.add(rest)
        self.assertEqual(len(index), len(first) + len(rest))
        # Postings must cover the documents added in both passes.
        for doc_ids in index.postings.values():
            self.assertEqual(len(doc_ids), len(set(doc_ids)))
        self.assertTrue(index.search("samba og", QuerySpec(in_stock_only=False)))

    def test_results_match_with_and_without_numpy(self):
        index = fixture_index()
        with_numpy = [h.listing.listing_id
                      for h in index.search("air max 90", QuerySpec(in_stock_only=False))]
        original = retrieve_module._np
        try:
            retrieve_module._np = None
            index._reindex()
            without = [h.listing.listing_id
                       for h in index.search("air max 90", QuerySpec(in_stock_only=False))]
        finally:
            retrieve_module._np = original
        self.assertEqual(with_numpy, without)

    def test_rerank_depth_caps_the_candidate_pool(self):
        index = fixture_index()
        hits = index.search("shoes", QuerySpec(in_stock_only=False), top_k=999)
        self.assertLessEqual(len(hits), max(RERANK_DEPTH, len(index)))

    def test_empty_index_is_safe(self):
        self.assertEqual(VectorIndex().search("air max 90"), [])


class TestAgentNearMiss(unittest.TestCase):
    def test_model_gap_is_reported(self):
        from sneakerrag.agent import model_gap
        products = cluster_listings(
            [l for a in get_sources() for l in a.search("air max", limit=999)])
        air_max_90 = next(p for p in products if p.model == "air max 90")
        spec = QuerySpec(terms="air max 95")
        self.assertIn("95", model_gap(spec, air_max_90))
        self.assertEqual(model_gap(QuerySpec(terms="air max 90"), air_max_90), "")


if __name__ == "__main__":
    unittest.main()

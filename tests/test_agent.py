import tempfile
import unittest
from pathlib import Path

from sneakerrag.agent import SneakerAgent, heuristic_spec, render_template_answer
from sneakerrag.store import Catalog
from tests.helpers import air_max_90_across_sites, listing


class TestQueryUnderstanding(unittest.TestCase):
    def test_brand_size_and_budget(self):
        spec = heuristic_spec("cheapest nike air max 90 in size 10.5 under $120")
        self.assertEqual(spec.brands, ["nike"])
        self.assertEqual(spec.size, "10.5")
        self.assertEqual(spec.max_price, 120.0)
        self.assertEqual(spec.terms, "air max 90")

    def test_air_max_is_not_read_as_a_budget(self):
        spec = heuristic_spec("cheapest nike air max 90")
        self.assertIsNone(spec.max_price)
        self.assertEqual(spec.terms, "air max 90")

    def test_model_number_is_not_mistaken_for_a_shoe_size(self):
        self.assertEqual(heuristic_spec("air max 90 size 10.5").size, "10.5")
        self.assertEqual(heuristic_spec("samba og in a 10").size, "10")
        self.assertEqual(heuristic_spec("990v6 grey").size, "")

    def test_style_code_lookup(self):
        spec = heuristic_spec("who has DD1391-100 cheapest")
        self.assertEqual(spec.style_code, "DD1391-100")
        self.assertEqual(spec.intent, "lookup")

    def test_deals_intent_and_discount(self):
        spec = heuristic_spec("show me adidas sales at least 20% off")
        self.assertEqual(spec.intent, "deals")
        self.assertEqual(spec.min_discount, 20.0)
        self.assertEqual(spec.brands, ["adidas"])

    def test_shipping_preference_and_gender(self):
        spec = heuristic_spec("women's 990v6 excluding shipping")
        self.assertFalse(spec.include_shipping)
        self.assertEqual(spec.gender, "women")

    def test_budget_number_is_not_left_in_the_model_terms(self):
        spec = heuristic_spec("New Balance 990v6 under $180 including shipping")
        self.assertEqual(spec.terms, "990v6")


class TestAgentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        catalog = Catalog(Path(cls.tmp.name) / "agent.db")
        catalog.upsert_listings(air_max_90_across_sites() + [
            listing("footlocker", "adidas Samba OG - Cloud White/Core Black", 99.99,
                    style_code="B75806", list_price=100.0),
            listing("jdsports", "adidas Samba OG Unisex", 84.99, shipping=6.99,
                    style_code="B75806", list_price=100.0),
            listing("adidas", "Samba OG Shoes Cloud White/Core Black", 100.0,
                    style_code="B75806"),
            listing("newbalance", "New Balance 990v6 Men's Grey", 199.99,
                    style_code="M990GL6", sizes=["9", "10"]),
            listing("dickssportinggoods", "New Balance Men's 990v6 Grey", 179.99,
                    style_code="M990GL6", in_stock=False),
        ])
        cls.catalog = catalog
        cls.agent = SneakerAgent(catalog=catalog, use_llm=False)

    @classmethod
    def tearDownClass(cls):
        cls.catalog.close()
        cls.tmp.cleanup()

    def test_cheapest_listing_wins_on_delivered_price(self):
        spec, products = self.agent.compare("cheapest adidas samba og", limit=1)
        best = products[0].best_offer()
        self.assertEqual(best.source, "jdsports")            # 84.99 + 6.99 = 91.98
        self.assertEqual(products[0].best_offer(include_shipping=False).source, "jdsports")

    def test_shipping_can_flip_the_winner(self):
        spec, products = self.agent.compare("adidas samba og", limit=1)
        product = products[0]
        expensive_shipping = [l for l in product.listings if l.source == "jdsports"][0]
        expensive_shipping.shipping = 20.0                   # 84.99 + 20 = 104.99
        self.assertEqual(product.best_offer().source, "footlocker")

    def test_all_six_style_code_siblings_are_grouped(self):
        _, products = self.agent.compare("samba", limit=1)
        self.assertEqual(len({l.source for l in products[0].listings}), 3)

    def test_out_of_stock_is_excluded_but_reported(self):
        answer = self.agent.answer("cheapest new balance 990v6", limit=1)
        self.assertIn("New Balance", answer.text)
        self.assertIn("excluded", answer.text.lower())
        self.assertIn("out of stock", answer.text.lower())

    def test_answer_cites_every_listing(self):
        answer = self.agent.answer("cheapest air max 90", limit=1)
        self.assertEqual(len(answer.citations), 4)
        self.assertEqual([c.n for c in answer.citations], [1, 2, 3, 4])
        for citation in answer.citations:
            self.assertTrue(citation.url.startswith("http"))

    def test_budget_that_cannot_be_met_is_stated(self):
        answer = self.agent.answer("air max 90 under $50", limit=1)
        self.assertIn("Nothing under $50.00", answer.text)

    def test_size_filter_removes_listings_without_that_size(self):
        spec, products = self.agent.compare("990v6 size 10.5", limit=1)
        self.assertEqual(spec.size, "10.5")
        self.assertEqual(products[0].offers(size="10.5", in_stock_only=True), [])

    def test_unknown_shoe_returns_a_graceful_answer(self):
        answer = self.agent.answer("cheapest asics gel kayano 31")
        self.assertEqual(answer.generator, "template")
        self.assertTrue(answer.text)

    def test_json_serialization_round_trips(self):
        payload = self.agent.answer("cheapest air max 90", limit=1).to_json()
        self.assertIn("citations", payload)
        self.assertIn("air max 90", payload.lower())

    def test_template_answer_names_the_winner_and_the_saving(self):
        spec, products = self.agent.compare("air max 90", limit=1)
        context, citations = self.agent.build_context(products, spec)
        text = render_template_answer("air max 90", spec, products, citations)
        self.assertIn("Cheapest:", text)
        self.assertIn("Saves", text)


if __name__ == "__main__":
    unittest.main()


class StubLLM:
    """Stands in for Claude so the LLM branches are testable without a key."""

    name = "stub"
    available = True

    def __init__(self, parse_json: str = "{}", answer: str = "stub answer [1]") -> None:
        self.parse_json = parse_json
        self.answer_text = answer
        self.calls: list[str] = []

    def complete(self, system: str, user: str, **kw) -> str:
        self.calls.append(user)
        return self.parse_json if "Question:" in user else self.answer_text


class FailingLLM(StubLLM):
    def complete(self, system: str, user: str, **kw) -> str:
        from sneakerrag.llm import LLMUnavailable
        raise LLMUnavailable("boom")


class TestClaudeBranch(unittest.TestCase):
    """Exercises the Claude path with a stub, including its failure modes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog = Catalog(Path(cls.tmp.name) / "llm.db")
        cls.catalog.upsert_listings(air_max_90_across_sites())

    @classmethod
    def tearDownClass(cls):
        cls.catalog.close()
        cls.tmp.cleanup()

    def agent(self, llm):
        return SneakerAgent(catalog=self.catalog, llm=llm)

    def test_llm_fills_in_what_the_regexes_missed(self):
        llm = StubLLM(parse_json='{"brands": ["nike"], "terms": "air max 90", '
                                 '"size": "11", "intent": "compare"}')
        spec = self.agent(llm).understand("that white nike runner everyone wears")
        self.assertEqual(spec.brands, ["nike"])
        self.assertEqual(spec.size, "11")

    def test_heuristics_win_over_the_llm_for_explicit_values(self):
        llm = StubLLM(parse_json='{"size": "8", "max_price": 999}')
        spec = self.agent(llm).understand("air max 90 size 10.5 under $120")
        self.assertEqual(spec.size, "10.5")
        self.assertEqual(spec.max_price, 120.0)

    def test_malformed_llm_json_is_ignored(self):
        spec = self.agent(StubLLM(parse_json="not json at all")).understand("air max 90")
        self.assertEqual(spec.terms, "air max 90")

    def test_answer_uses_claude_when_available(self):
        answer = self.agent(StubLLM()).answer("cheapest air max 90", limit=1)
        self.assertEqual(answer.generator, "claude")
        self.assertEqual(answer.text, "stub answer [1]")
        self.assertTrue(answer.citations)

    def test_answer_falls_back_to_template_when_claude_fails(self):
        answer = self.agent(FailingLLM()).answer("cheapest air max 90", limit=1)
        self.assertEqual(answer.generator, "template")
        self.assertIn("Cheapest:", answer.text)

    def test_context_block_is_numbered_and_carries_urls(self):
        agent = self.agent(StubLLM())
        spec, products = agent.compare("air max 90", limit=1)
        context, citations = agent.build_context(products, spec)
        for citation in citations:
            self.assertIn(f"[{citation.n}]", context)
            self.assertIn(citation.url, context)

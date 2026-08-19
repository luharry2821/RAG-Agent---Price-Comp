import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sneakerrag.cli import main


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "cli.db")
        cls.invoke("ingest")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def invoke(cls, *args: str) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--db", cls.db, *args])
        return code, buf.getvalue()

    def test_ingest_loaded_all_six_sites(self):
        code, out = self.invoke("stats")
        self.assertEqual(code, 0)
        stats = json.loads(out)
        self.assertEqual(len(stats["by_source"]), 6)
        self.assertGreaterEqual(stats["listings"], 30)

    def test_sources_listing(self):
        code, out = self.invoke("sources")
        self.assertEqual(code, 0)
        self.assertIn("footlocker", out)
        self.assertIn("New Balance", out)

    def test_compare_marks_a_cheapest_listing(self):
        code, out = self.invoke("compare", "air", "max", "90", "--no-color")
        self.assertEqual(code, 0)
        self.assertIn("CHEAPEST", out)

    def test_compare_json_is_machine_readable(self):
        code, out = self.invoke("compare", "samba", "--json", "--limit", "1")
        payload = json.loads(out)
        self.assertEqual(payload["query"], "samba")
        self.assertTrue(payload["products"][0]["listings"])

    def test_ask_offline_produces_citations(self):
        code, out = self.invoke("ask", "cheapest", "990v6", "--no-llm", "--limit", "1")
        self.assertEqual(code, 0)
        self.assertIn("Sources", out)
        self.assertIn("[1]", out)
        self.assertIn("offline template", out)

    def test_ask_json_shape(self):
        code, out = self.invoke("ask", "cheapest", "samba", "--no-llm", "--json", "--limit", "1")
        payload = json.loads(out)
        self.assertIn("answer", payload)
        self.assertTrue(payload["citations"])

    def test_product_by_style_code(self):
        code, out = self.invoke("product", "B75806")
        self.assertEqual(code, 0)
        self.assertIn("Samba", out)

    def test_unknown_style_code_exits_nonzero(self):
        code, _ = self.invoke("product", "ZZ0000-000")
        self.assertEqual(code, 1)

    def test_deals_lists_discounts(self):
        code, out = self.invoke("deals", "--min-pct", "5", "--limit", "5")
        self.assertEqual(code, 0)
        self.assertIn("%", out)

    def test_reingest_is_idempotent(self):
        _, before = self.invoke("stats")
        self.invoke("ingest")
        _, after = self.invoke("stats")
        self.assertEqual(json.loads(before)["listings"], json.loads(after)["listings"])


if __name__ == "__main__":
    unittest.main()

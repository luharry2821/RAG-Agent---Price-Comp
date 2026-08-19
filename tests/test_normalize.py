import unittest

from sneakerrag.normalize import (
    detect_gender, extract_style_code, normalize_brand, normalize_colorway,
    normalize_size, parse_price, parse_title,
)


class TestBrand(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_brand("Nike Sportswear"), "nike")
        self.assertEqual(normalize_brand("adidas Originals"), "adidas")
        self.assertEqual(normalize_brand("NEW BALANCE"), "new balance")
        self.assertEqual(normalize_brand("Air Jordan"), "nike")
        self.assertEqual(normalize_brand("Asics Gel-Kayano"), "")


class TestStyleCodes(unittest.TestCase):
    def test_nike_formats(self):
        self.assertEqual(extract_style_code("Dunk Low DD1391-100", "nike"), "DD1391-100")
        self.assertEqual(extract_style_code("Air Force 1 315122-111", "nike"), "315122-111")

    def test_adidas_and_new_balance(self):
        self.assertEqual(extract_style_code("Samba OG B75806", "adidas"), "B75806")
        self.assertEqual(extract_style_code("Ultraboost IE1766", "adidas"), "IE1766")
        self.assertEqual(extract_style_code("990v6 M990GL6", "new balance"), "M990GL6")

    def test_no_false_positive_on_plain_model_number(self):
        self.assertEqual(extract_style_code("New Balance 990", "new balance"), "")
        self.assertEqual(extract_style_code("Air Max 90", "nike"), "")


class TestTitles(unittest.TestCase):
    def test_gendered_marketing_title(self):
        parsed = parse_title("Nike Air Max 90 Men's Shoes")
        self.assertEqual(parsed["brand"], "nike")
        self.assertEqual(parsed["model"], "air max 90")
        self.assertEqual(parsed["gender"], "men")

    def test_separator_title_with_colorway(self):
        parsed = parse_title("Nike Air Max 90 - Men's - White/Black")
        self.assertEqual(parsed["model"], "air max 90")
        self.assertEqual(parsed["colorway"], "white / black")

    def test_trailing_colorway_without_separator(self):
        parsed = parse_title("adidas Originals Ultraboost 5 Running Shoes Core Black")
        self.assertEqual(parsed["model"], "ultraboost 5")
        self.assertEqual(parsed["colorway"], "core black")

    def test_brand_word_not_eaten_by_noise_filter(self):
        parsed = parse_title("New Balance 990v6 Made in USA Men's Running Shoes M990GL6 - Grey")
        self.assertEqual(parsed["brand"], "new balance")
        self.assertIn("990v6", parsed["model"])
        self.assertNotIn("balance", parsed["model"])
        self.assertEqual(parsed["style_code"], "M990GL6")

    def test_colorway_with_slash_is_not_split(self):
        parsed = parse_title("adidas Samba OG - Cloud White / Core Black")
        self.assertEqual(parsed["model"], "samba og")
        self.assertEqual(parsed["colorway"], "cloud white / core black")


class TestScalars(unittest.TestCase):
    def test_prices(self):
        self.assertEqual(parse_price("$129.99"), (129.99, "USD"))
        self.assertEqual(parse_price("USD 1,299.00"), (1299.0, "USD"))
        self.assertEqual(parse_price("£99.95"), (99.95, "GBP"))
        self.assertEqual(parse_price(84.5), (84.5, "USD"))
        self.assertEqual(parse_price("")[0], None)

    def test_sizes(self):
        self.assertEqual(normalize_size("US M 10.5"), "10.5")
        self.assertEqual(normalize_size(9), "9")
        self.assertEqual(normalize_size("10 1/2"), "10.5")

    def test_gender(self):
        self.assertEqual(detect_gender("Women's Running Shoe"), "women")
        self.assertEqual(detect_gender("Grade School"), "kids")

    def test_colorway_dedupe(self):
        self.assertEqual(normalize_colorway("White/White/Black"), "white / black")


if __name__ == "__main__":
    unittest.main()

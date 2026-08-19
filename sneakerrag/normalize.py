"""Text normalization: brands, models, colorways, style codes, prices, sizes.

Retailers describe the same shoe very differently::

    "Nike Air Max 90 Men's Shoes"
    "Air Max 90 - Men's - White/Black"
    "Nike Air Max 90 (HM0089-100) White Black"

Everything downstream (matching, retrieval, comparison) depends on squeezing
those into a comparable shape, so this module is deliberately conservative:
it only strips things it is sure about.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Brands
# --------------------------------------------------------------------------

BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "nike": ("nike", "nike sportswear", "nikelab", "jordan", "air jordan", "nike acg"),
    "adidas": ("adidas", "adidas originals", "adidas performance", "adidas terrex", "y-3"),
    "new balance": ("new balance", "newbalance", "nb", "new balance numeric"),
}

# Sub-labels that should keep their parent brand but not pollute the model name.
BRAND_SUBLABELS = ("originals", "performance", "sportswear", "nikelab", "terrex", "numeric")

SUPPORTED_BRANDS = tuple(BRAND_ALIASES)


def normalize_brand(text: str) -> str:
    """Return the canonical brand for a title/brand string, or ''."""
    t = f" {squash(text)} "
    # Longest alias first so "new balance numeric" wins over "new balance".
    for brand, aliases in BRAND_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if f" {alias} " in t:
                return brand
    return ""


# --------------------------------------------------------------------------
# Style codes (manufacturer SKUs)
# --------------------------------------------------------------------------
#
# These are the single most reliable join key between retailers, so we detect
# them per-brand rather than with one loose pattern.

STYLE_CODE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    # DD1391-100, HM0089-100 (2 letters + 4 digits) and legacy 315122-111.
    "nike": (
        re.compile(r"\b([A-Z]{2}\d{4}-\d{3})\b"),
        re.compile(r"\b(\d{6}-\d{3})\b"),
    ),
    # IE1766, GY7386, B75806, FX5502 — 1-2 letters + 4-5 digits.
    "adidas": (
        re.compile(r"\b([A-Z]{2}\d{4})\b"),
        re.compile(r"\b([A-Z]\d{5})\b"),
    ),
    # M990GL6, U574LGVC, BB550PB1, ML574EVG, MFC(...) — letters, digits, letters.
    "new balance": (
        re.compile(r"\b([A-Z]{1,3}\d{3,4}[A-Z]{1,4}\d{0,2})\b"),
    ),
}

_ANY_STYLE_CODE = re.compile(
    r"\b([A-Z]{2}\d{4}-\d{3}|\d{6}-\d{3}|[A-Z]{2}\d{4}|[A-Z]\d{5}|[A-Z]{1,3}\d{3,4}[A-Z]{1,4}\d{0,2})\b"
)

# Words that look like style codes but are not (sizes, years, model numbers).
_CODE_STOPWORDS = {"UK", "US", "EU", "GS", "TD", "PS", "OG", "SE", "QS"}


def extract_style_code(text: str, brand: str = "") -> str:
    """Pull a manufacturer style code out of free text.

    Passing ``brand`` narrows the pattern set and greatly reduces false hits.
    """
    if not text:
        return ""
    upper = text.upper()
    patterns = STYLE_CODE_PATTERNS.get(brand, ())
    for pattern in patterns or (_ANY_STYLE_CODE,):
        for m in pattern.finditer(upper):
            code = m.group(1)
            if code in _CODE_STOPWORDS:
                continue
            # A pure model number such as "990" or "574" is not a style code.
            if code.isdigit():
                continue
            return code
    return ""


def normalize_style_code(code: str) -> str:
    """Canonical form: uppercase, no spaces, single hyphen."""
    if not code:
        return ""
    c = re.sub(r"[\s_]+", "", code.upper())
    c = re.sub(r"-{2,}", "-", c)
    return c.strip("-")


# --------------------------------------------------------------------------
# Titles -> model / colorway
# --------------------------------------------------------------------------

_NOISE_WORDS = (
    "shoes", "shoe", "sneakers", "sneaker", "trainers", "trainer", "footwear",
    "running", "casual", "lifestyle", "mens", "men", "womens", "women",
    "unisex", "adults", "adult", "grade school", "big kids", "little kids",
    "kids", "junior", "juniors", "gs", "sale", "new", "exclusive", "online",
    "buy", "official", "store", "free shipping",
)

_NOISE_PHRASES = (
    "made in usa", "made in the usa", "made in uk", "made in england",
    "grade school", "big kids", "little kids", "free shipping", "online only",
    "limited edition", "official site",
)

_GENDER_PATTERNS = (
    (re.compile(r"\b(women'?s|womens|female|wmns)\b", re.I), "women"),
    (re.compile(r"\b(men'?s|mens|male)\b", re.I), "men"),
    (re.compile(r"\b(kids?|junior|grade school|toddler|gs|ps|td)\b", re.I), "kids"),
    (re.compile(r"\bunisex\b", re.I), "unisex"),
)

_SEPARATORS = re.compile(r"\s+[-–—]\s+|\s*\|\s*")


def detect_gender(text: str) -> str:
    for pattern, gender in _GENDER_PATTERNS:
        if pattern.search(text or ""):
            return gender
    return ""


def squash(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[\u2019']s\b", "", t)      # men's -> men, not "men s"
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9./\s-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def strip_noise(text: str) -> str:
    """Remove marketing/gender/category words that retailers bolt on."""
    t = squash(text)
    for phrase in _NOISE_PHRASES:
        t = t.replace(phrase, " ")
    for pattern, _ in _GENDER_PATTERNS:
        t = pattern.sub(" ", t)
    for word in sorted(_NOISE_WORDS, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(word)}\b", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" -/")


def parse_title(title: str, brand_hint: str = "") -> dict[str, str]:
    """Split a retailer title into brand / model / colorway / style code."""
    raw = title or ""
    brand = normalize_brand(brand_hint) or normalize_brand(raw)
    style_code = normalize_style_code(extract_style_code(raw, brand))
    gender = detect_gender(raw)

    body = raw
    if style_code:
        body = re.sub(re.escape(style_code), " ", body, flags=re.I)
    body = re.sub(r"[()\[\]]", " ", body)

    # Colorways usually follow the last " - " / " | " separator and contain a
    # slash ("White/Black") or are clearly colour words.
    parts = [p.strip() for p in _SEPARATORS.split(body) if p.strip()]
    colorway = ""
    if len(parts) > 1 and _looks_like_colorway(parts[-1]):
        colorway = parts.pop()
    head = " ".join(parts) if parts else body

    # Strip the brand *before* the noise pass: "new" is a noise word, and
    # removing it first would turn "New Balance 990v6" into "balance 990v6".
    head = squash(head)
    if brand:
        for alias in sorted(BRAND_ALIASES[brand], key=len, reverse=True):
            head = re.sub(rf"\b{re.escape(alias)}\b", " ", head)
        for sub in BRAND_SUBLABELS:
            head = re.sub(rf"\b{re.escape(sub)}\b", " ", head)
    model = strip_noise(head)
    model = " ".join(w for w in model.split() if len(w) > 1 or w.isdigit())

    if not colorway:
        model, colorway = _split_trailing_colorway(model)
    model = re.sub(r"\s+", " ", model).strip(" -/")

    return {
        "brand": brand,
        "model": model,
        "colorway": normalize_colorway(colorway),
        "style_code": style_code,
        "gender": gender,
    }


_COLOR_WORDS = {
    "white", "black", "grey", "gray", "red", "blue", "green", "yellow", "pink",
    "purple", "orange", "brown", "beige", "cream", "navy", "olive", "sail",
    "gum", "silver", "gold", "bone", "sand", "teal", "burgundy", "maroon",
    "ivory", "charcoal", "khaki", "platinum", "crimson", "infrared", "volt",
    "triple", "core", "cloud", "chalk", "wolf", "sea", "salt", "onyx", "mint",
}


def _looks_like_colorway(text: str) -> bool:
    t = squash(text)
    if not t:
        return False
    tokens = set(re.split(r"[\s/]+", t))
    if tokens & _COLOR_WORDS:
        return True
    return "/" in t and len(t) <= 40


def _split_trailing_colorway(model: str) -> tuple[str, str]:
    """Move a trailing run of colour words out of the model name.

    "air max 90 white/black" -> ("air max 90", "white / black").
    Only fires when something recognisable is left behind, so a model that is
    *only* colour words (rare, e.g. "gum") survives intact.
    """
    tokens = model.split()
    tail: list[str] = []
    while tokens:
        token = tokens[-1]
        pieces = [p for p in token.split("/") if p]
        if pieces and all(p in _COLOR_WORDS for p in pieces):
            tail.insert(0, tokens.pop())
        else:
            break
    if not tail or not tokens:
        return model, ""
    return " ".join(tokens), normalize_colorway(" ".join(tail))


def normalize_colorway(text: str) -> str:
    """Canonical colorway: lowercase, ' / ' separated, deduped."""
    t = squash(text)
    if not t:
        return ""
    t = re.sub(r"\b(colou?r|colorway)\b", " ", t)
    parts = [p.strip() for p in re.split(r"[/,]| and ", t) if p.strip()]
    seen: list[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p)
        if p and p not in seen:
            seen.append(p)
    return " / ".join(seen)


# --------------------------------------------------------------------------
# Prices and sizes
# --------------------------------------------------------------------------

_CURRENCY_BY_SYMBOL = {"$": "USD", "£": "GBP", "€": "EUR", "US$": "USD", "CA$": "CAD"}
_PRICE_RE = re.compile(r"(?P<sym>US\$|CA\$|[$£€])?\s*(?P<num>\d[\d,]*(?:\.\d{1,2})?)")


def parse_price(text: str | float | int | None) -> tuple[float | None, str]:
    """Parse '$129.99', 'USD 129.99', 129.99 -> (129.99, 'USD')."""
    if text is None or text == "":
        return None, "USD"
    if isinstance(text, (int, float)):
        return round(float(text), 2), "USD"
    s = str(text).strip()
    currency = "USD"
    upper = s.upper()
    for code in ("USD", "GBP", "EUR", "CAD"):
        if code in upper:
            currency = code
            break
    m = _PRICE_RE.search(s)
    if not m:
        return None, currency
    if m.group("sym"):
        currency = _CURRENCY_BY_SYMBOL.get(m.group("sym"), currency)
    try:
        return round(float(m.group("num").replace(",", "")), 2), currency
    except ValueError:
        return None, currency


_SIZE_RE = re.compile(r"(?:us\s*)?(?:m|w)?\s*(\d{1,2}(?:\.5)?)", re.I)


def normalize_size(size: str | float | int) -> str:
    """'US M 10.5' / 10.5 / '10 1/2' -> '10.5'."""
    if isinstance(size, (int, float)):
        return f"{float(size):g}"
    s = str(size or "").strip().replace("½", ".5").replace(" 1/2", ".5")
    m = _SIZE_RE.search(s)
    if not m:
        return s.lower()
    return f"{float(m.group(1)):g}"


def normalize_sizes(sizes) -> list[str]:
    out: list[str] = []
    for s in sizes or []:
        n = normalize_size(s)
        if n and n not in out:
            out.append(n)
    return out


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", squash(text)).strip("-")

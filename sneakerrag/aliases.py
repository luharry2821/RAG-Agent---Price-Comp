"""Sneaker vocabulary: nicknames, abbreviations and colorway slang.

Shoppers do not type catalogue titles. They type "panda dunks", "AF1s",
"sambas", "ultrabost". No embedding model fixes that reliably — "panda" is
semantically a bear — but a small domain dictionary does, and it is the single
highest-value retrieval improvement for this corpus.

Expansion is additive and damped: the original token always keeps full weight,
so an alias can add recall but cannot outvote what the user actually wrote.
"""

from __future__ import annotations

import re

from .normalize import squash

ALIAS_WEIGHT = 0.9      # aliases count slightly less than the typed word
PLURAL_WEIGHT = 0.85    # "sambas" -> "samba"
FUZZY_WEIGHT = 0.6      # typo repair is the weakest signal

# Nickname / abbreviation -> the words that appear in real listing titles.
NICKNAMES: dict[str, str] = {
    # Nike
    "af1": "air force 1",
    "af1s": "air force 1",
    "am90": "air max 90",
    "am95": "air max 95",
    "am97": "air max 97",
    "am1": "air max 1",
    "panda": "dunk low white black",
    "pandas": "dunk low white black",
    "dunks": "dunk",
    "sb": "dunk",
    "pegasus": "pegasus",
    "vapormax": "air vapormax",
    "tns": "air max plus",
    "tn": "air max plus",
    # adidas
    "ub": "ultraboost",
    "ubs": "ultraboost",
    "sambas": "samba",
    "gazelles": "gazelle",
    "campus 00": "campus 00s",
    "spezials": "spezial",
    "yeezys": "yeezy",
    "supers tar": "superstar",
    # New Balance
    "nb": "new balance",
    "990s": "990",
    "550s": "550",
    "574s": "574",
    "1906": "1906r",
    "992s": "992",
    "dad": "990 1906r 2002r",
    "dad shoe": "990 1906r 2002r",
    "dad shoes": "990 1906r 2002r",
    # generic
    "og": "og",
    "gs": "grade school",
    "ds": "deadstock",
}

# Common misspellings that a trigram matcher handles less reliably than a map.
MISSPELLINGS: dict[str, str] = {
    "adiddas": "adidas",
    "addidas": "adidas",
    "adidias": "adidas",
    "balence": "balance",
    "ballance": "balance",
    "newbalance": "new balance",
    "nikes": "nike",
    "jordons": "jordan",
    "ultrabost": "ultraboost",
    "ultraboots": "ultraboost",
    "peagasus": "pegasus",
    "colourway": "colorway",
}

# Colorway slang -> the colour words retailers actually print.
COLOR_SLANG: dict[str, str] = {
    "panda": "white black",
    "oreo": "white black",
    "triple white": "white",
    "triple black": "black",
    "bred": "black red",
    "unc": "university blue white",
    "chicago": "red white black",
    "sail": "sail cream",
    "gum": "gum brown",
}

# Words that appear in half the catalogue or carry no product identity. IDF
# already discounts them; removing them outright stops them from crowding out
# the one token that actually identifies the shoe ("dunks" in "nike dunks
# size 13"). Applied to queries only — the index keeps every word.
QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "at", "to", "with",
    "me", "my", "i", "is", "are", "im", "id", "please", "can", "you",
    "shoes", "shoe", "sneakers", "sneaker", "trainers", "trainer", "kicks",
    "creps", "footwear", "pair", "pairs",
    "cheapest", "cheap", "cheaper", "best", "lowest", "price", "prices",
    "cost", "costs", "deal", "deals", "buy", "get", "find", "show", "give",
    "size", "sizes", "us", "uk", "available", "stock", "site", "sites",
    "website", "websites", "seller", "sellers", "where", "which", "what",
    "who", "how", "much", "right", "now", "currently", "under", "below",
})

_TOKEN = re.compile(r"[a-z0-9.]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(squash(text))


def meaningful_tokens(text: str) -> list[str]:
    """The shopper's own content words, before any expansion.

    Empty when the query is nothing but noise ("cheapest shoes") — which the
    index reads as "browse whatever the filters allow" rather than as a failed
    search.
    """
    return [t for t in _tokens(text) if t not in QUERY_STOPWORDS]


def _expand(text: str) -> dict[str, tuple[float, set[str]]]:
    """Weighted search terms, each tagged with the typed word(s) it came from.

    Tracking provenance matters: an expansion must not be able to vouch for
    itself. "chicago" expanding to "white black" should mark *chicago* as
    satisfied when those colours match — not add two independent matches, which
    would let the dictionary manufacture the relevance it is being judged on.
    """
    out: dict[str, tuple[float, set[str]]] = {}

    def add(term: str, weight: float, source: str) -> None:
        if not term:
            return
        current_weight, sources = out.get(term, (0.0, set()))
        out[term] = (max(current_weight, weight), sources | {source})

    lowered = squash(text)
    # Multi-word keys first ("dad shoes" must beat "shoes").
    for phrase, expansion in sorted(
        {**NICKNAMES, **COLOR_SLANG}.items(), key=lambda kv: -len(kv[0])
    ):
        if " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", lowered):
            for token in _tokens(expansion):
                add(token, ALIAS_WEIGHT, phrase.split()[0])

    # If the shopper typed nothing but noise, search the noise rather than
    # returning an empty query.
    for token in (meaningful_tokens(text) or _tokens(text)):
        add(token, 1.0, token)
        if token in MISSPELLINGS:
            for fixed in _tokens(MISSPELLINGS[token]):
                add(fixed, ALIAS_WEIGHT, token)
        if token in NICKNAMES:
            for alias in _tokens(NICKNAMES[token]):
                add(alias, ALIAS_WEIGHT, token)
        if token in COLOR_SLANG:
            for colour in _tokens(COLOR_SLANG[token]):
                add(colour, ALIAS_WEIGHT, token)
        # "sambas" -> "samba", "dunks" -> "dunk"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            add(token[:-1], PLURAL_WEIGHT, token)

    return out


def expand_query(text: str) -> list[tuple[str, float]]:
    """Weighted search terms: the shopper's own words at 1.0, additions below."""
    return sorted(((term, weight) for term, (weight, _src) in _expand(text).items()),
                  key=lambda kv: (-kv[1], kv[0]))


def term_sources(text: str) -> dict[str, set[str]]:
    """Which typed word each search term came from."""
    return {term: sources for term, (_weight, sources) in _expand(text).items()}


def trigrams(term: str) -> set[str]:
    padded = f"  {term} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def similarity(a: str, b: str) -> float:
    """Trigram Jaccard — cheap, and good at catching single-character typos."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

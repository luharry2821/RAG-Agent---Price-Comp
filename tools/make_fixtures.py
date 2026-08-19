#!/usr/bin/env python3
"""Generate the offline sample catalogue in data/fixtures/.

The listings are SYNTHETIC: realistic in shape (title styles, style-code
presence, promo pricing, stock and size gaps differ per retailer) but the
prices are invented. They exist so the pipeline, matcher and CLI can be run and
tested without hitting any retailer. Run with ``python3 tools/make_fixtures.py``.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path("data/fixtures")
SEED = 20260819

# brand, model, colorway, style code, MSRP, gender, silhouette sizes
CATALOG = [
    ("Nike", "Air Max 90", "White/Black", "HM0089-100", 130.0, "men"),
    ("Nike", "Air Force 1 '07", "Triple White", "CW2288-111", 115.0, "men"),
    ("Nike", "Dunk Low Retro", "White/Black", "DD1391-100", 120.0, "men"),
    ("Nike", "Pegasus 41", "Black/White", "FD2723-001", 140.0, "women"),
    ("adidas", "Samba OG", "Cloud White/Core Black", "B75806", 100.0, "unisex"),
    ("adidas", "Ultraboost 5", "Core Black", "IE1766", 190.0, "men"),
    ("adidas", "Campus 00s", "Grey/White", "HQ8708", 110.0, "unisex"),
    ("New Balance", "990v6", "Grey", "M990GL6", 200.0, "men"),
    ("New Balance", "550", "White/Green", "BB550PB1", 120.0, "unisex"),
    ("New Balance", "Fresh Foam X 1080v13", "Black", "M1080B13", 165.0, "men"),
]

SIZES = ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12", "13"]

# key -> (title template, shows style code, shipping, price policy)
SITE_STYLE = {
    "nike":       ("{brand} {model} {gender_word} Shoes",              True,  0.0,  (1.00, 1.00)),
    "adidas":     ("{model} Shoes",                                    True,  0.0,  (0.95, 1.00)),
    "newbalance": ("{model} {gender_word}",                            True,  0.0,  (0.97, 1.00)),
    "footlocker": ("{brand} {model} - {gender_word} - {colorway}",     True,  0.0,  (0.80, 1.00)),
    "jdsports":   ("{brand} {model} {gender_word}",                    False, 6.99, (0.75, 0.98)),
    "dickssportinggoods": ("{brand} {gender_word} {model} Shoes",      False, 0.0,  (0.85, 1.05)),
}

BRAND_SITES = {"Nike": "nike", "adidas": "adidas", "New Balance": "newbalance"}
MULTI_SITES = ["footlocker", "jdsports", "dickssportinggoods"]

GENDER_WORD = {"men": "Men's", "women": "Women's", "unisex": "Unisex", "kids": "Kids'"}


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-").replace("--", "-")


def product_url(site: str, brand: str, model: str, code: str, idx: int) -> str:
    s = slugify(f"{brand} {model}")
    return {
        "nike": f"https://www.nike.com/t/{s}-{code[:6].lower()}/{code}",
        "adidas": f"https://www.adidas.com/us/{s}/{code}.html",
        "newbalance": f"https://www.newbalance.com/pd/{s}/{code}.html",
        "footlocker": f"https://www.footlocker.com/product/~{s}/{code}.html",
        "jdsports": f"https://www.jdsports.com/product/{s}/{16000000 + idx}/",
        "dickssportinggoods": f"https://www.dickssportinggoods.com/p/{s}/{22000000 + idx}",
    }[site]


def build() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    now = datetime.now(timezone.utc)
    out: dict[str, list[dict]] = {key: [] for key in SITE_STYLE}

    for idx, (brand, model, colorway, code, msrp, gender) in enumerate(CATALOG):
        carriers = [BRAND_SITES[brand]]
        # Multi-brand retailers carry most, but not all, of the range.
        for site in MULTI_SITES:
            if rng.random() < 0.82:
                carriers.append(site)

        for site in carriers:
            template, shows_code, shipping, (lo, hi) = SITE_STYLE[site]
            factor = round(rng.uniform(lo, hi), 3)
            price = round(msrp * factor, 2)
            price = round(price - 0.01 if price % 1 == 0 and site not in BRAND_SITES.values() else price, 2)
            on_sale = price < msrp * 0.97
            sizes = [s for s in SIZES if rng.random() < (0.85 if site in BRAND_SITES.values() else 0.7)]
            in_stock = bool(sizes) and rng.random() > 0.08

            title = template.format(brand=brand, model=model, colorway=colorway,
                                    gender_word=GENDER_WORD[gender]).replace("  ", " ").strip()
            if shows_code and rng.random() < 0.55:
                title = f"{title} ({code})"

            record = {
                "title": title,
                "url": product_url(site, brand, model, code, idx),
                "brand": brand,
                "colorway": colorway,
                "style_code": code if shows_code else "",
                "retailer_sku": f"{site[:3].upper()}-{100000 + idx * 7 + len(out[site])}",
                "price": price,
                "list_price": msrp if on_sale else None,
                "currency": "USD",
                "shipping": shipping,
                "in_stock": in_stock,
                "sizes": sizes,
                "gender": gender,
                "image": f"https://images.example.com/{slugify(brand + ' ' + model)}.jpg",
                "scraped_at": (now - timedelta(hours=rng.randint(0, 20))).isoformat(timespec="seconds"),
            }
            out[site].append(record)

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build()
    for site, listings in data.items():
        payload = {
            "_note": "SYNTHETIC sample data generated by tools/make_fixtures.py. "
                     "Prices are invented and do not reflect real retailer pricing.",
            "source": site,
            "listings": listings,
        }
        (OUT / f"{site}.json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
        print(f"{site:22} {len(listings):3d} listings")


if __name__ == "__main__":
    main()

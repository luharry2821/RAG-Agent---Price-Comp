#!/usr/bin/env python3
"""Generate the offline sample catalogue in data/fixtures/.

The listings are SYNTHETIC: realistic in *shape* — brand stores at one price
across sizes, resale marketplaces with per-size ask curves, fees, mixed
condition and sold-out sizes — but every price is invented. They let the whole
pipeline run and be tested without touching a retailer.

Run with ``python3 tools/make_fixtures.py``.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path("data/fixtures")
SEED = 20260820

# brand, model, colorway, style code, MSRP, gender, hype, still sold at retail
#
# "hype" drives the resale premium: 0 = sits at or below retail, 1 = every size
# trades well above MSRP. Shoes flagged not-at-retail are sold out at the brand
# store and exist only on the resale sites, which is the case those sites are
# actually for.
CATALOG = [
    ("Nike", "Air Max 90", "White/Black", "HM0089-100", 130.0, "men", 0.10, True),
    ("Nike", "Air Force 1 '07", "Triple White", "CW2288-111", 115.0, "men", 0.05, True),
    ("Nike", "Dunk Low Retro", "White/Black", "DD1391-100", 120.0, "men", 0.75, False),
    ("Nike", "Pegasus 41", "Black/White", "FD2723-001", 140.0, "women", 0.00, True),
    ("adidas", "Samba OG", "Cloud White/Core Black", "B75806", 100.0, "unisex", 0.55, True),
    ("adidas", "Ultraboost 5", "Core Black", "IE1766", 190.0, "men", 0.00, True),
    ("adidas", "Campus 00s", "Grey/White", "HQ8708", 110.0, "unisex", 0.35, True),
    ("New Balance", "990v6", "Grey", "M990GL6", 200.0, "men", 0.30, True),
    ("New Balance", "550", "White/Green", "BB550PB1", 120.0, "unisex", 0.20, True),
    ("New Balance", "1906R", "Silver/Sea Salt", "M1906RA", 165.0, "unisex", 0.60, False),
]

SIZES = ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12", "13"]

BRAND_SITE = {"Nike": "nike", "adidas": "adidas", "New Balance": "newbalance"}

# key -> title template, shows style code, shipping, fees, ask multiplier range
RESALE_SITES = {
    "stadiumgoods": ("{brand} {model} '{colorway}'",            0.35, 12.00, 0.00, (1.02, 1.18)),
    "flightclub":   ("{brand} {model} {colorway}",              0.30, 14.50, 0.00, (1.00, 1.14)),
    "goat":         ("{brand} {model} '{colorway}' ({code})",   0.85, 12.00, 5.00, (0.96, 1.10)),
    "kickscrew":    ("{brand} {model} {colorway} {code}",       0.95,  0.00, 0.00, (0.90, 1.04)),
}

BRAND_TEMPLATE = {
    "nike": "{brand} {model} {gender_word} Shoes",
    "adidas": "{model} Shoes",
    "newbalance": "{model} {gender_word}",
}

GENDER_WORD = {"men": "Men's", "women": "Women's", "unisex": "Unisex", "kids": "Kids'"}


def slugify(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def product_url(site: str, brand: str, model: str, colorway: str, code: str, idx: int) -> str:
    shoe = slugify(f"{brand} {model}")
    full = slugify(f"{brand} {model} {colorway}")
    return {
        "nike": f"https://www.nike.com/t/{shoe}-{code[:6].lower()}/{code}",
        "adidas": f"https://www.adidas.com/us/{shoe}/{code}.html",
        "newbalance": f"https://www.newbalance.com/pd/{shoe}/{code}.html",
        "stadiumgoods": f"https://www.stadiumgoods.com/en-us/shopping/{full}-{code.lower()}",
        "flightclub": f"https://www.flightclub.com/{full}-{code.lower()}",
        "goat": f"https://www.goat.com/sneakers/{full}-{code.lower()}",
        "kickscrew": f"https://www.kickscrew.com/products/{full}-{code.lower()}",
    }[site]


def size_curve(msrp: float, hype: float, base_factor: float, rng: random.Random,
               sizes: list[str]) -> dict[str, float]:
    """Per-size asks: core sizes carry the premium, edge sizes sit cheapest."""
    prices: dict[str, float] = {}
    for size in sizes:
        value = float(size)
        # Bell around a US 10 — where demand (and therefore the ask) peaks.
        demand = max(0.0, 1.0 - abs(value - 10.0) / 4.0)
        factor = base_factor * (1.0 + hype * (0.45 * demand - 0.10))
        factor *= rng.uniform(0.97, 1.03)
        prices[size] = round(msrp * factor, 2)
    return prices


def build() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    now = datetime.now(timezone.utc)
    out: dict[str, list[dict]] = {key: [] for key in list(BRAND_TEMPLATE) + list(RESALE_SITES)}

    for idx, (brand, model, colorway, code, msrp, gender, hype, at_retail) in enumerate(CATALOG):
        # ---- brand store ------------------------------------------------
        if at_retail:
            site = BRAND_SITE[brand]
            sizes = [s for s in SIZES if rng.random() < 0.8]
            on_sale = hype < 0.2 and rng.random() < 0.4
            price = round(msrp * (rng.uniform(0.78, 0.92) if on_sale else 1.0), 2)
            out[site].append({
                "title": BRAND_TEMPLATE[site].format(
                    brand=brand, model=model, gender_word=GENDER_WORD[gender]).strip(),
                "url": product_url(site, brand, model, colorway, code, idx),
                "brand": brand, "colorway": colorway, "style_code": code,
                "retailer_sku": code,
                "price": price, "list_price": msrp if on_sale else None,
                "currency": "USD", "shipping": 0.0, "fees": 0.0,
                "in_stock": bool(sizes), "sizes": sizes, "size_prices": {},
                "condition": "new", "gender": gender,
                "image": f"https://images.example.com/{slugify(brand + ' ' + model)}.jpg",
                "scraped_at": (now - timedelta(hours=rng.randint(0, 20))).isoformat(timespec="seconds"),
            })

        # ---- resale marketplaces ---------------------------------------
        for site, (template, code_odds, shipping, fees, span) in RESALE_SITES.items():
            if rng.random() > (0.95 if not at_retail else 0.85):
                continue                                   # not every site has every shoe
            stocked = [s for s in SIZES if rng.random() < 0.6]
            if not stocked:
                continue
            base_factor = rng.uniform(*span) * (1.0 + hype * 0.35)
            prices = size_curve(msrp, hype, base_factor, rng, stocked)
            used = site in ("goat", "flightclub") and rng.random() < 0.25
            if used:
                prices = {s: round(p * rng.uniform(0.68, 0.82), 2) for s, p in prices.items()}

            title = template.format(brand=brand, model=model, colorway=colorway,
                                    code=code if rng.random() < code_odds else "").strip()
            title = " ".join(title.split()).replace(" ()", "")
            out[site].append({
                "title": title,
                "url": product_url(site, brand, model, colorway, code, idx),
                "brand": brand, "colorway": colorway,
                "style_code": code if rng.random() < code_odds else "",
                "retailer_sku": f"{site[:3].upper()}-{200000 + idx * 11 + len(out[site])}",
                "price": min(prices.values()), "list_price": msrp,
                "currency": "USD", "shipping": shipping, "fees": fees,
                "in_stock": True, "sizes": stocked, "size_prices": prices,
                "condition": "used" if used else "new", "gender": gender,
                "image": f"https://images.example.com/{slugify(brand + ' ' + model)}.jpg",
                "scraped_at": (now - timedelta(hours=rng.randint(0, 20))).isoformat(timespec="seconds"),
            })

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for site, listings in build().items():
        payload = {
            "_note": "SYNTHETIC sample data generated by tools/make_fixtures.py. Prices are "
                     "invented and do not reflect real retailer or marketplace pricing.",
            "source": site,
            "listings": listings,
        }
        (OUT / f"{site}.json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
        print(f"{site:16} {len(listings):3d} listings")


if __name__ == "__main__":
    main()

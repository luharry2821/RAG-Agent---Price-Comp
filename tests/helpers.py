"""Shared fixtures for the test-suite."""

from __future__ import annotations

from sneakerrag.models import Listing
from sneakerrag.normalize import parse_title

SITE_KIND = {"nike": "brand", "adidas": "brand", "newbalance": "brand"}
SITE_NAME = {
    "nike": "Nike", "adidas": "adidas", "newbalance": "New Balance",
    "stadiumgoods": "Stadium Goods", "flightclub": "Flight Club",
    "goat": "GOAT", "kickscrew": "KicksCrew",
}


def listing(source: str, title: str, price: float, *, style_code: str = "",
            shipping: float | None = 0.0, fees: float | None = None,
            in_stock: bool = True, sizes: list[str] | None = None,
            size_prices: dict[str, float] | None = None, condition: str = "new",
            list_price: float | None = None, url: str = "") -> Listing:
    parsed = parse_title(title)
    return Listing(
        source=source,
        source_name=SITE_NAME.get(source, source.title()),
        source_kind=SITE_KIND.get(source, "resale"),
        url=url or f"https://{source}.example/p/{abs(hash((source, title, price))) % 10**8}",
        title=title,
        brand=parsed["brand"],
        model=parsed["model"],
        colorway=parsed["colorway"],
        style_code=style_code or parsed["style_code"],
        gender=parsed["gender"],
        price=price,
        list_price=list_price,
        shipping=shipping,
        fees=fees,
        in_stock=in_stock,
        condition=condition,
        sizes=sizes if sizes is not None else ["9", "10", "10.5", "11"],
        size_prices=size_prices or {},
    )


def air_max_90_across_sites() -> list[Listing]:
    """One shoe, four sellers: a brand store plus three resale marketplaces.

    Deliberately inconsistent titles, and the resale sellers price per size.
    """
    return [
        listing("nike", "Nike Air Max 90 Men's Shoes (HM0089-100)", 130.00),
        listing("stadiumgoods", "Nike Air Max 90 'White/Black'", 119.99, list_price=130.0,
                shipping=12.0, size_prices={"9": 119.99, "10": 132.0, "10.5": 141.0, "11": 128.0}),
        listing("goat", "Nike Air Max 90 'White/Black' (HM0089-100)", 109.99, shipping=12.0,
                fees=5.0, list_price=130.0,
                size_prices={"9": 109.99, "10": 124.0, "10.5": 118.0, "11": 121.0}),
        listing("kickscrew", "Nike Air Max 90 White/Black HM0089-100", 114.99, shipping=0.0,
                list_price=130.0, sizes=["8", "9", "11"],
                size_prices={"8": 114.99, "9": 121.0, "11": 133.0}),
    ]

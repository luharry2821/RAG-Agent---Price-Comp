"""Shared fixtures for the test-suite."""

from __future__ import annotations

from sneakerrag.models import Listing
from sneakerrag.normalize import parse_title


def listing(source: str, title: str, price: float, *, style_code: str = "",
            shipping: float | None = 0.0, in_stock: bool = True,
            sizes: list[str] | None = None, list_price: float | None = None,
            url: str = "") -> Listing:
    parsed = parse_title(title)
    return Listing(
        source=source,
        source_name=source.replace("_", " ").title(),
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
        in_stock=in_stock,
        sizes=sizes if sizes is not None else ["9", "10", "10.5", "11"],
    )


def air_max_90_across_sites() -> list[Listing]:
    """One shoe, four retailers, deliberately inconsistent titles."""
    return [
        listing("nike", "Nike Air Max 90 Men's Shoes (HM0089-100)", 130.00),
        listing("footlocker", "Nike Air Max 90 - Men's - White/Black", 119.99, list_price=130.0),
        listing("jdsports", "Nike Air Max 90 Men's", 109.99, shipping=6.99, list_price=130.0),
        listing("dickssportinggoods", "Nike Men's Air Max 90 Shoes", 114.99,
                sizes=["8", "9", "11"]),
    ]

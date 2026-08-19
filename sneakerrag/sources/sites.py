"""The retailer set: three brand stores plus three multi-brand retailers.

Brand stores anchor the MSRP; multi-brand retailers are where discounts show
up. That mix is what makes a cross-site comparison worth running.
"""

from __future__ import annotations

from .base import SiteSpec

SITES: tuple[SiteSpec, ...] = (
    SiteSpec(
        key="nike",
        name="Nike",
        home="https://www.nike.com/",
        search_url="https://www.nike.com/w?q={q}",
        product_link=r"/t/[a-z0-9-]+",
        brands=("nike",),
        notes="Brand store. Style code appears on the product page as 'Style: XXXXXX-XXX'.",
    ),
    SiteSpec(
        key="adidas",
        name="adidas",
        home="https://www.adidas.com/",
        search_url="https://www.adidas.com/us/search?q={q}",
        product_link=r"/us/[a-z0-9-]+/[A-Z]{2}\d{4}\.html",
        brands=("adidas",),
        notes="Brand store. The product code is embedded in the URL slug.",
    ),
    SiteSpec(
        key="newbalance",
        name="New Balance",
        home="https://www.newbalance.com/",
        search_url="https://www.newbalance.com/search?q={q}",
        product_link=r"/pd/[a-z0-9-]+/[A-Z0-9]+\.html",
        brands=("new balance",),
        notes="Brand store. Style code is the trailing path segment.",
    ),
    SiteSpec(
        key="footlocker",
        name="Foot Locker",
        home="https://www.footlocker.com/",
        search_url="https://www.footlocker.com/search?query={q}",
        product_link=r"/product/[~a-z0-9-]+/\w+\.html",
        notes="Multi-brand. Frequent promo pricing; JSON-LD includes offers.",
    ),
    SiteSpec(
        key="jdsports",
        name="JD Sports",
        home="https://www.jdsports.com/",
        search_url="https://www.jdsports.com/search/{q}/",
        product_link=r"/product/[a-z0-9-]+/\d+",
        notes="Multi-brand. Often the cheapest on carry-over colourways.",
    ),
    SiteSpec(
        key="dickssportinggoods",
        name="Dick's Sporting Goods",
        home="https://www.dickssportinggoods.com/",
        search_url="https://www.dickssportinggoods.com/search/SearchDisplay?searchTerm={q}",
        product_link=r"/p/[a-z0-9-]+/\w+",
        notes="Multi-brand. Ships free above a threshold; shipping modelled per listing.",
    ),
)

SITES_BY_KEY = {s.key: s for s in SITES}


def sites_for_brand(brand: str) -> list[SiteSpec]:
    return [s for s in SITES if s.sells(brand)]

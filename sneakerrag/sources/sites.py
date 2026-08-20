"""The site set: three brand stores plus four resale marketplaces.

The two kinds behave very differently and the pipeline has to know which is
which:

* **brand** stores (nike.com, adidas.com, newbalance.com) sell at one price for
  every size, in new condition, and are the MSRP anchor.
* **resale** marketplaces (Stadium Goods, Flight Club, GOAT, KicksCrew) price
  *per size* from seller asks, mix new and used, and add shipping plus
  authentication/processing fees. A size 8 being cheap there says nothing about
  what a size 12 costs.

Note that Flight Club and GOAT are both GOAT Group properties and often show
overlapping inventory — treat a match between them as one supply pool rather
than two independent quotes.
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
        kind="brand",
        notes="Brand store. Style code shown on the product page as 'Style: XXXXXX-XXX'. "
              "One price across sizes; free shipping for members.",
    ),
    SiteSpec(
        key="adidas",
        name="adidas",
        home="https://www.adidas.com/",
        search_url="https://www.adidas.com/us/search?q={q}",
        product_link=r"/us/[a-z0-9-]+/[A-Z]{2}\d{4}\.html",
        brands=("adidas",),
        kind="brand",
        notes="Brand store. The product code is embedded in the URL slug.",
    ),
    SiteSpec(
        key="newbalance",
        name="New Balance",
        home="https://www.newbalance.com/",
        search_url="https://www.newbalance.com/search?q={q}",
        product_link=r"/pd/[a-z0-9-]+/[A-Z0-9]+\.html",
        brands=("new balance",),
        kind="brand",
        notes="Brand store. Style code is the trailing path segment.",
    ),
    SiteSpec(
        key="stadiumgoods",
        name="Stadium Goods",
        home="https://www.stadiumgoods.com/",
        search_url="https://www.stadiumgoods.com/en-us/search?q={q}",
        product_link=r"/en-us/shopping/[a-z0-9-]+",
        kind="resale",
        notes="Consignment. Per-size asks, mostly deadstock; carries sizes long gone "
              "from the brand stores, at a premium.",
    ),
    SiteSpec(
        key="flightclub",
        name="Flight Club",
        home="https://www.flightclub.com/",
        search_url="https://www.flightclub.com/catalogsearch/result?query={q}",
        product_link=r"/[a-z0-9]+(?:-[a-z0-9]+){2,}$",
        kind="resale",
        notes="Consignment (GOAT Group). Per-size asks; new and used listed side by side. "
              "Inventory overlaps GOAT.",
    ),
    SiteSpec(
        key="goat",
        name="GOAT",
        home="https://www.goat.com/",
        search_url="https://www.goat.com/search?query={q}",
        product_link=r"/sneakers/[a-z0-9-]+",
        kind="resale",
        notes="Marketplace (GOAT Group). Per-size lowest ask, new/used/defect tiers, "
              "plus shipping and a processing fee at checkout.",
    ),
    SiteSpec(
        key="kickscrew",
        name="KicksCrew",
        home="https://www.kickscrew.com/",
        search_url="https://www.kickscrew.com/en-US/search?q={q}",
        product_link=r"/products/[a-z0-9-]+",
        kind="resale",
        notes="Marketplace sourcing from global sellers. Often the cheapest ask, with "
              "longer international shipping times.",
    ),
)

SITES_BY_KEY = {s.key: s for s in SITES}
BRAND_SITES = tuple(s for s in SITES if s.kind == "brand")
RESALE_SITES = tuple(s for s in SITES if s.kind == "resale")


def sites_for_brand(brand: str) -> list[SiteSpec]:
    return [s for s in SITES if s.sells(brand)]

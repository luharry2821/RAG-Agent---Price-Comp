"""sneakerrag — RAG price comparison for Nike, adidas and New Balance sneakers."""

from .agent import SneakerAgent
from .models import Answer, Citation, Listing, Product, QuerySpec
from .store import Catalog

__version__ = "0.1.0"
__all__ = ["SneakerAgent", "Catalog", "Listing", "Product", "QuerySpec", "Answer", "Citation"]

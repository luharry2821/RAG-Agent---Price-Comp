"""Source registry: fixtures by default, live scraping opt-in."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .base import LiveSource, SiteSpec, SourceAdapter, StaticSource, build_listing
from .http import HttpClient
from .sites import SITES, SITES_BY_KEY, sites_for_brand

FIXTURE_DIR = Path(os.environ.get("SNEAKERRAG_FIXTURES", "data/fixtures"))


def load_fixture(spec: SiteSpec, fixture_dir: Path | None = None) -> StaticSource:
    path = (fixture_dir or FIXTURE_DIR) / f"{spec.key}.json"
    records = []
    if path.exists():
        payload = json.loads(path.read_text("utf-8"))
        records = payload.get("listings", payload) if isinstance(payload, dict) else payload
    return StaticSource(spec, records)


def get_sources(keys: Iterable[str] | None = None, *, mode: str = "",
                client: HttpClient | None = None,
                fixture_dir: Path | None = None) -> list[SourceAdapter]:
    """Build adapters for the requested site keys.

    ``mode`` is ``fixtures`` (default, offline sample data) or ``live``
    (real HTTP fetches, robots-aware).
    """
    mode = (mode or os.environ.get("SNEAKERRAG_MODE", "fixtures")).lower()
    wanted = list(keys) if keys else list(SITES_BY_KEY)
    adapters: list[SourceAdapter] = []
    for key in wanted:
        spec = SITES_BY_KEY.get(key)
        if spec is None:
            raise KeyError(f"unknown source '{key}'; known: {', '.join(SITES_BY_KEY)}")
        adapters.append(LiveSource(spec, client=client) if mode == "live"
                        else load_fixture(spec, fixture_dir))
    return adapters


__all__ = [
    "SITES", "SITES_BY_KEY", "SiteSpec", "SourceAdapter", "StaticSource",
    "LiveSource", "HttpClient", "build_listing", "get_sources", "load_fixture",
    "sites_for_brand", "FIXTURE_DIR",
]

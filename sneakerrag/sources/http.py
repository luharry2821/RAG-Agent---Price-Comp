"""A deliberately polite HTTP client for the live source adapters.

Retail sites are third-party property. This client therefore:

* honours ``robots.txt`` for the configured user agent (opt out only with an
  explicit flag, e.g. when you have written permission or an API agreement);
* rate-limits per host with a configurable minimum interval and jitter;
* caches responses on disk so re-running ``ingest`` does not re-hit the site;
* identifies itself honestly in the User-Agent, with a contact URL.

It uses :mod:`requests` when available and falls back to :mod:`urllib`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

DEFAULT_UA = os.environ.get(
    "SNEAKERRAG_USER_AGENT",
    "sneakerrag/0.1 (+https://github.com/luharry2821/RAG-Agent---Price-Comp; price research bot)",
)
DEFAULT_CACHE = Path(os.environ.get("SNEAKERRAG_CACHE", ".cache/http"))


class FetchError(RuntimeError):
    pass


class RobotsDisallowed(FetchError):
    pass


@dataclass
class HttpClient:
    user_agent: str = DEFAULT_UA
    timeout: float = 20.0
    min_interval: float = 2.0          # seconds between requests to one host
    max_retries: int = 3
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE)
    cache_ttl: float = 6 * 3600
    obey_robots: bool = True
    _last_hit: dict[str, float] = field(default_factory=dict)
    _robots: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)

    # -- robots -----------------------------------------------------------
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        host = urllib.parse.urlsplit(url)._replace(path="/robots.txt", query="", fragment="")
        origin = f"{host.scheme}://{host.netloc}"
        if origin in self._robots:
            return self._robots[origin]
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(urllib.parse.urlunsplit(host))
        try:
            parser.read()
        except Exception:
            # Unreadable robots.txt: treat as "no rules published" but keep the
            # rate limiter in charge.
            parser = urllib.robotparser.RobotFileParser()
            parser.parse([])
        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.obey_robots:
            return True
        parser = self._robots_for(url)
        try:
            return bool(parser and parser.can_fetch(self.user_agent, url))
        except Exception:
            return False

    # -- cache ------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _cached(self, url: str) -> str | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        if self.cache_ttl and time.time() - path.stat().st_mtime > self.cache_ttl:
            return None
        try:
            return json.loads(path.read_text("utf-8"))["body"]
        except Exception:
            return None

    def _store(self, url: str, body: str) -> None:
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "fetched_at": time.time(), "body": body}), "utf-8")

    # -- fetch ------------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait + random.uniform(0, 0.4))
        self._last_hit[host] = time.time()

    def get(self, url: str, *, use_cache: bool = True) -> str:
        if use_cache:
            cached = self._cached(url)
            if cached is not None:
                return cached
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url} for {self.user_agent}")

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(url)
            try:
                if requests is not None:
                    resp = requests.get(url, headers=headers, timeout=self.timeout)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise FetchError(f"HTTP {resp.status_code}")
                    resp.raise_for_status()
                    body = resp.text
                else:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                        body = fh.read().decode(fh.headers.get_content_charset() or "utf-8", "replace")
                if use_cache:
                    self._store(url, body)
                return body
            except Exception as exc:            # retry with exponential backoff
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 0.5))
        raise FetchError(f"failed to fetch {url}: {last_error}")

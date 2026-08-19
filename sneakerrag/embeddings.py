"""Dependency-free text embeddings for the vector half of retrieval.

Anthropic does not serve an embeddings endpoint, and a sneaker catalogue is
short-text data where lexical signal dominates — so the default embedder here
is a deterministic hashed bag-of-features (word unigrams + bigrams + character
4-grams) rather than a neural model. It needs no network, no model download and
no API key, which keeps ``ingest`` fast and reproducible.

If you want dense semantic embeddings, implement :class:`Embedder` and pass it
to the index; :func:`get_embedder` will pick up ``sentence-transformers`` when
it is installed and ``SNEAKERRAG_EMBEDDER=st`` is set.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Iterable, Protocol, Sequence

DEFAULT_DIM = 512


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


def _features(text: str) -> Iterable[tuple[str, float]]:
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    words = t.split()
    feats: dict[str, float] = {}

    def bump(key: str, weight: float) -> None:
        feats[key] = feats.get(key, 0.0) + weight

    for w in words:
        bump(f"w:{w}", 1.0)
        if any(ch.isdigit() for ch in w):
            bump(f"n:{w}", 1.5)          # model numbers carry extra weight
    for a, b in zip(words, words[1:]):
        bump(f"b:{a}_{b}", 0.8)
    padded = f" {t} "
    for i in range(len(padded) - 3):
        bump(f"c:{padded[i:i + 4]}", 0.35)   # robust to typos/spacing
    return feats.items()


def _hash(key: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0        # signed hashing cancels collisions
    return (value >> 1) % dim, sign


class HashingEmbedder:
    """Signed feature hashing with sublinear term frequency and L2 norm."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for key, weight in _features(text):
            idx, sign = _hash(key, self.dim)
            vec[idx] += sign * weight * (1.0 + math.log(1.0 + weight))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:
    """Optional dense embedder (only used when the package is installed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def get_embedder(kind: str = "") -> Embedder:
    kind = kind or os.environ.get("SNEAKERRAG_EMBEDDER", "hashing")
    if kind in ("st", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:                    # not installed / no model cache
            pass
    return HashingEmbedder()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; inputs from this module are already unit length."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)

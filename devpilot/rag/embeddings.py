"""Turn text into vectors.

An embedding model maps text to a point in space where *meaning* is distance: "how do I
log in" lands near "authentication setup" even with no shared words. Two implementations:

* ``APIEmbedder``  -- any OpenAI-compatible /embeddings endpoint (OpenRouter by default).
* ``HashEmbedder`` -- deterministic bag-of-words hashing. Not semantic, but it needs no
  network, so tests and offline demos can exercise the whole pipeline.
"""

import math
import re
import zlib
from typing import Any, Protocol


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class APIEmbedder:
    BATCH = 64

    def __init__(self, client: Any, model: str):
        self.client = client
        self.name = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = [t[:8000] for t in texts[i : i + self.BATCH]]  # stay under model input limits
            resp = await self.client.embeddings.create(model=self.name, input=batch)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vectors


class HashEmbedder:
    name = "hash-256"
    DIMS = 256

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = zlib.crc32(word.encode())
            vec[h % self.DIMS] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

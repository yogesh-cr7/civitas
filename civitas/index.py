"""A tiny persisted vector index: build once, search many times.

For 139 chunks, brute-force cosine similarity over a dense numpy matrix is
faster than the disk I/O needed to load a proper vector database, so that's
all this does. It won't scale to a million-document corpus, but pretending
otherwise for a demo corpus this size would be the wrong kind of over-engineering.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import SentenceTransformerEmbedder, TfidfEmbedder, get_embedder
from .ingest import Chunk, load_corpus

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index.json"

_EMBEDDER_CLASSES = {
    "tfidf": TfidfEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def build(cls, embedder_name: str = "tfidf", split_long: bool = True) -> "VectorIndex":
        chunks = load_corpus(split_long=split_long)
        embedder = get_embedder(embedder_name)
        embedder.fit([c.text for c in chunks])
        vectors = embedder.encode([c.text for c in chunks])
        return cls(chunks, vectors, embedder)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vec = self.embedder.encode([query])[0]
        # vectors are L2-normalized, so the dot product is cosine similarity
        scores = self.vectors @ query_vec
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder_name": self.embedder.name,
            "embedder_state": self.embedder.to_dict(),
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": self.vectors.tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_PATH) -> "VectorIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        embedder_cls = _EMBEDDER_CLASSES[data["embedder_name"]]
        embedder = embedder_cls.from_dict(data["embedder_state"])
        chunks = [Chunk(**c) for c in data["chunks"]]
        vectors = np.array(data["vectors"], dtype=np.float32)
        return cls(chunks, vectors, embedder)

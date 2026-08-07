"""Embedding backends for turning chunk text into vectors.

Two backends, picked by the caller:

- "tfidf" (default): a small, dependency-free (numpy only) TF-IDF vectorizer.
  No download, no API key, works instantly on a 139-chunk corpus. This is
  the honest choice for a corpus this size -- semantic embeddings are not
  obviously better here, and that comparison is exactly the kind of thing
  the eval suite is meant to measure rather than assume.
- "sentence-transformers": real dense semantic embeddings via the
  sentence-transformers library, for anyone who wants to install the extra
  ~500MB of dependencies and compare retrieval quality against TF-IDF.

Both implement the same tiny interface: fit(texts) then encode(texts).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfEmbedder:
    """Minimal TF-IDF vectorizer with cosine-ready L2-normalized output."""

    name = "tfidf"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        doc_freq: Counter[str] = Counter()
        tokenized = [set(_tokenize(t)) for t in texts]
        for tokens in tokenized:
            doc_freq.update(tokens)

        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, df in doc_freq.items():
            # smoothed idf, same shape as sklearn's default
            self.idf[self.vocab[term]] = math.log((1 + n_docs) / (1 + df)) + 1
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(_tokenize(text))
            if not counts:
                continue
            max_count = max(counts.values())
            for term, count in counts.items():
                col = self.vocab.get(term)
                if col is None:
                    continue  # out-of-vocabulary token (query-time only)
                tf = 0.5 + 0.5 * (count / max_count)  # smoothed term frequency
                vectors[row, col] = tf * self.idf[col]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "idf": self.idf.tolist() if self.idf is not None else None}

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfEmbedder":
        emb = cls()
        emb.vocab = data["vocab"]
        emb.idf = np.array(data["idf"], dtype=np.float32) if data["idf"] is not None else None
        return emb


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Imported lazily so it's never a hard dependency."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` or use --embedder tfidf."
            ) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained model, nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"model_name": self.model_name}

    @classmethod
    def from_dict(cls, data: dict) -> "SentenceTransformerEmbedder":
        return cls(model_name=data["model_name"])


def get_embedder(name: str = "tfidf"):
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder: {name!r} (choices: tfidf, sentence-transformers)")

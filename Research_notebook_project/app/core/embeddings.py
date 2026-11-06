# app/core/embeddings.py
"""
Local text embedding provider for CiteGuard using bge-small (384 dimensions).
Reused across Phase 1 research chunking and Phase 2 document ingestion.
"""
import hashlib
import math
from collections.abc import Sequence

EMBEDDING_DIM = 384
_model = None


def get_embedding_model():
    """Lazy loader for SentenceTransformer to minimize startup latency."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception:
            _model = None
    return _model


def _deterministic_fallback_embedding(text: str) -> list[float]:
    """
    Deterministic 384-dimensional unit vector generation for test environments
    or when heavy weights are still loading.
    """
    vec = [0.0] * EMBEDDING_DIM
    words = text.lower().split()
    if not words:
        vec[0] = 1.0
        return vec

    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(EMBEDDING_DIM):
            byte_val = digest[i % len(digest)]
            # Map byte (0-255) to float (-1.0 to 1.0)
            val = (byte_val / 127.5) - 1.0
            vec[i] += val

    # L2 Normalization
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0

    return vec


def embed_text(text: str) -> list[float]:
    """Computes a 384-dimensional normalized embedding for a single text."""
    model = get_embedding_model()
    if model is not None:
        try:
            vector = model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        except Exception:
            pass

    return _deterministic_fallback_embedding(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Batch computes 384-dimensional normalized embeddings."""
    if not texts:
        return []

    model = get_embedding_model()
    if model is not None:
        try:
            vectors = model.encode(list(texts), normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        except Exception:
            pass

    return [_deterministic_fallback_embedding(t) for t in texts]

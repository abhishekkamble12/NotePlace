# app/core/embeddings.py
"""
Local text embeddings with BAAI/bge-small-en-v1.5 (384 dimensions, normalized), used for chunk storage and
retrieval. If the model cannot load, this raises: there is no fake fallback vector.
"""
from collections.abc import Sequence

EMBEDDING_DIM = 384
_model = None


def get_embedding_model():
    """Lazy loader so API startup does not pay the model load."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def embed_text(text: str) -> list[float]:
    """384-dimensional normalized embedding for one text."""
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Batch 384-dimensional normalized embeddings."""
    if not texts:
        return []
    return [v.tolist() for v in get_embedding_model().encode(list(texts), normalize_embeddings=True)]

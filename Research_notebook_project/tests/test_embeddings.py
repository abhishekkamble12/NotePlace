"""Real bge-small model (downloads ~130 MB on first run). No fallback vectors exist, so this is the model itself."""
from app.core.embeddings import embed_text, embed_texts


def test_real_model_ranks_related_text_closer():
    q = embed_text("perovskite solar cell efficiency record")
    related, unrelated = embed_texts([
        "The new perovskite photovoltaic cell reached 33.9% efficiency.",
        "Bake the chocolate cake for 30 minutes at 180 degrees.",
    ])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731 - vectors are normalized, so dot = cosine
    assert len(q) == 384
    assert abs(dot(q, q) - 1.0) < 1e-3
    assert dot(q, related) > dot(q, unrelated) + 0.1


def test_empty_batch_needs_no_model():
    assert embed_texts([]) == []

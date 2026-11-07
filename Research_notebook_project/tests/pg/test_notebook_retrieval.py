"""Notebook ingestion + hybrid retrieval on real Postgres + pgvector: offsets, pages, metadata, idempotency, tenant isolation."""
import pytest
from sqlalchemy import select, text

from app.ingest.pdf import PdfError
from app.ingest.service import create_notebook, get_or_create_user, ingest_pdf
from app.models import Chunk, Source
from app.retrieval import MODES, retrieve, rrf, tsquery_text
from tests.ingest.pdfmaker import make_pdf

pytestmark = pytest.mark.int


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    """The test PDFs are a few sentences: shrink the chunk limit so each paragraph becomes its own chunk and ranking,
    section and page assertions are non-trivial (with the real 480-token limit each PDF would be a single chunk)."""
    from app.ingest import chunking
    monkeypatch.setattr(chunking, "MAX_TOKENS", 30)

GEO = make_pdf([
    [("Volcanoes", 18), ("Magma forms in the mantle where rock partially melts, and rises through the crust to feed volcanoes.", 10)],
    [("Glaciers", 18), ("Glaciers store fresh water as ice and retreat when the climate warms. Zxqvortex is a rare marker term.", 10)],
])
BIO = make_pdf([
    [("Photosynthesis", 18), ("Chlorophyll in the chloroplast absorbs sunlight to convert carbon dioxide and water into glucose.", 10)],
])


async def _setup(sm, chunkers=("structure",)):
    async with sm() as s:
        a, b = await get_or_create_user(s, "user-a"), await get_or_create_user(s, "user-b")
        na, nb = await create_notebook(s, a.id, "A"), await create_notebook(s, b.id, "B")
        sa, _ = await ingest_pdf(s, a.id, na.id, GEO, "geo.pdf", chunkers)
        sb, _ = await ingest_pdf(s, b.id, nb.id, GEO, "geo-copy.pdf", chunkers)  # same text in another tenant on purpose
        await ingest_pdf(s, b.id, nb.id, BIO, "bio.pdf", chunkers)
        await s.commit()
        return a.id, na.id, sa.id, b.id, nb.id, sb.id


@pytest.mark.asyncio
async def test_ingested_chunks_are_exact_snapshot_slices_with_derived_metadata(sm_factory):
    sm = sm_factory()
    ua, na, sid, *_ = await _setup(sm, ("fixed", "structure"))
    async with sm() as s:
        src = await s.get(Source, sid)
        assert src.job_id is None and src.kind == "pdf" and src.notebook_id == na and len(src.page_offsets) == 2
        chunks = (await s.execute(select(Chunk).where(Chunk.source_id == sid))).scalars().all()
        assert {c.chunker for c in chunks} == {"fixed", "structure"}
        for c in chunks:
            assert src.snapshot[c.start_char:c.end_char] == c.text and c.notebook_id == na and c.embedding is not None
        st = [c for c in chunks if c.chunker == "structure"]
        assert st[0].section == "Volcanoes" and st[0].page_start == 1
        glac = next(c for c in st if "Glaciers" in c.text)
        assert (glac.page_start, glac.page_end) == (2, 2) and src.snapshot.index("Glaciers") >= src.page_offsets[1]
        assert (await s.execute(text("SELECT count(*) FROM chunks WHERE tsv IS NOT NULL"))).scalar() == len(
            (await s.execute(select(Chunk.id))).scalars().all())
        idx = (await s.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_tsv'"))).scalar()
        assert idx and "gin" in idx.lower()


@pytest.mark.asyncio
async def test_reingest_is_idempotent_and_chunkers_are_added_separately(sm_factory):
    sm = sm_factory()
    async with sm() as s:
        u = await get_or_create_user(s, "u")
        nb = await create_notebook(s, u.id)
        src1, c1 = await ingest_pdf(s, u.id, nb.id, GEO, "g.pdf")
        n_structure = len((await s.execute(select(Chunk.id).where(Chunk.source_id == src1.id))).scalars().all())
        src2, c2 = await ingest_pdf(s, u.id, nb.id, GEO, "g-again.pdf")
        assert (c1, c2, src1.id == src2.id) == (True, False, True)
        assert len((await s.execute(select(Chunk.id).where(Chunk.source_id == src1.id))).scalars().all()) == n_structure
        await ingest_pdf(s, u.id, nb.id, GEO, "g.pdf", ("fixed", "structure"))
        by = (await s.execute(select(Chunk.chunker, Chunk.id).where(Chunk.source_id == src1.id))).all()
        assert sum(1 for k, _ in by if k == "structure") == n_structure and any(k == "fixed" for k, _ in by)


@pytest.mark.asyncio
async def test_cannot_ingest_into_another_users_notebook(sm_factory):
    sm = sm_factory()
    async with sm() as s:
        a, b = await get_or_create_user(s, "a"), await get_or_create_user(s, "b")
        nb = await create_notebook(s, a.id)
        with pytest.raises(LookupError):
            await ingest_pdf(s, b.id, nb.id, GEO, "x.pdf")
        with pytest.raises(PdfError):
            await ingest_pdf(s, a.id, nb.id, b"not a pdf", "x.pdf")
        assert (await s.execute(select(Source.id))).first() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tenant_isolation_every_mode_returns_only_own_notebook(sm_factory, mode):
    sm = sm_factory()
    ua, na, sa, ub, nb, sb = await _setup(sm)
    async with sm() as s:
        own_a = {c for c in (await s.execute(select(Chunk.id).where(Chunk.notebook_id == na))).scalars()}
        own_b = {c for c in (await s.execute(select(Chunk.id).where(Chunk.notebook_id == nb))).scalars()}
        assert own_a and own_b and not own_a & own_b
        for q in ("where does magma form", "zxqvortex", "chlorophyll sunlight"):
            ra = await retrieve(s, ua, na, q, mode)
            rb = await retrieve(s, ub, nb, q, mode)
            assert {h.chunk_id for h in ra.hits} <= own_a and {h.chunk_id for h in rb.hits} <= own_b
            # user B naming A's notebook, and A naming B's: nothing comes back (ownership checked in SQL)
            assert (await retrieve(s, ub, na, q, mode)).hits == []
            assert (await retrieve(s, ua, nb, q, mode)).hits == []
        # B's biology PDF is invisible to A even for an exact-term query
        assert all("Chlorophyll" not in h.text for h in (await retrieve(s, ua, na, "chlorophyll chloroplast", mode)).hits)


@pytest.mark.asyncio
async def test_missing_scope_is_an_error_not_a_global_search(sm_factory):
    sm = sm_factory()
    await _setup(sm)
    async with sm() as s:
        with pytest.raises(ValueError):
            await retrieve(s, None, 1, "magma")
        with pytest.raises(ValueError):
            await retrieve(s, 1, None, "magma")


@pytest.mark.asyncio
async def test_sparse_finds_rare_term_inside_natural_language_question(sm_factory):
    sm = sm_factory()
    ua, na, *_ = await _setup(sm)
    async with sm() as s:
        r = await retrieve(s, ua, na, "What is the marker term called zxqvortex in this document?", "hybrid")
        assert "Zxqvortex" in r.hits[0].text and set(r.timings_ms) == {"embed", "dense", "sparse", "fuse"}


@pytest.mark.asyncio
async def test_dense_finds_paraphrase_without_shared_words_and_hit_carries_citation_fields(sm_factory):
    sm = sm_factory()
    ua, na, sid, *_ = await _setup(sm)
    async with sm() as s:
        r = await retrieve(s, ua, na, "How does molten rock get to the surface?", "dense", k=3)
        top = r.hits[0]
        assert "Magma" in top.text and top.page_start == 1 and top.section == "Volcanoes"
        snap = (await s.get(Source, sid)).snapshot
        assert snap[top.start_char:top.end_char] == top.text  # citation by reference still resolves against the snapshot


@pytest.mark.asyncio
async def test_hybrid_rerank_orders_relevant_chunk_first_with_all_stage_timings(sm_factory):
    sm = sm_factory()
    ua, na, *_ = await _setup(sm, ("fixed", "structure"))
    async with sm() as s:
        r = await retrieve(s, ua, na, "What do glaciers do when the climate warms?", "hybrid+rerank", chunker="structure")
        assert "Glaciers" in r.hits[0].text and set(r.timings_ms) == {"embed", "dense", "sparse", "fuse", "rerank"}
        scores = [h.score for h in r.hits]
        assert scores == sorted(scores, reverse=True)
        fixed = await retrieve(s, ua, na, "What do glaciers do?", "dense", chunker="fixed")
        assert fixed.hits and all(h.section is None for h in fixed.hits)  # E1 chunks carry no structure


def test_rrf_hand_computed():
    out = dict(rrf([[1, 2, 3], [3, 1, 4]], k=60))
    assert out[1] == pytest.approx(1 / 61 + 1 / 62)
    assert out[3] == pytest.approx(1 / 63 + 1 / 61)
    assert out[2] == pytest.approx(1 / 62) and out[4] == pytest.approx(1 / 63)
    assert [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])] == [1, 3, 2, 4]
    assert rrf([]) == [] and rrf([[], []]) == []


def test_rrf_tie_breaks_by_id_deterministically():
    assert [i for i, _ in rrf([[5], [2]])] == [2, 5]


def test_tsquery_text_cannot_inject_operators():
    assert tsquery_text("a & !b | (c) ':* <-> d") == "a | b | c | d"
    assert tsquery_text("!!! ???") == ""
    assert tsquery_text("A a A") == "a"
    assert len(tsquery_text(" ".join(f"w{i}" for i in range(100))).split(" | ")) == 32

"""E1-E4 retrieval experiments on the QASPER gold (evals/qasper/gold.json). Writes evals/retrieval_results.json
(+ evals/retrieval_per_query.json). Local compute only: no LLM, no provider billing.

  E1  fixed 350-word chunks (existing chunker) + dense (pgvector)
  E2  structure-aware chunks + dense
  E3  structure-aware chunks + hybrid (dense + PostgreSQL full-text, RRF k=60)
  E4  E3 + cross-encoder rerank of the top-20 fused candidates

Setting: ALL papers are ingested into ONE notebook, so a question must find its evidence among every paper's chunks
(harder than the usual one-paper QASPER setting; the per-paper setting is not measured). Nothing is tuned on these
questions: candidates=20, RRF k=60, chunk limit 480 tokens, rerank model fixed in advance.
Relevance: a chunk covers a gold span if it is from the right paper and overlaps >= half of the shorter of the two
(scripts/eval_metrics.covers). Recall@k = share of a question's gold spans covered by the top-k chunks.
Latency is wall time per query on THIS machine with warm models, one run; it is not portable.

Needs a Postgres+pgvector server: --database-url or env EVAL_DATABASE_URL / TEST_DATABASE_URL (a throwaway database is
created and dropped). The local PDFs from `python scripts/build_qasper_gold.py` must exist and match the recorded hashes.
Usage: python scripts/run_retrieval_eval.py [--database-url postgresql+asyncpg://user:pw@host:5433/db]
"""
import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import covers, mean_ci, mrr_at, ndcg_at, paired_delta_ci, recall_covered  # noqa: E402

Q = ROOT / "evals" / "qasper"
OUT, PERQ = ROOT / "evals" / "retrieval_results.json", ROOT / "evals" / "retrieval_per_query.json"
EXPERIMENTS = {"E1": ("fixed", "dense"), "E2": ("structure", "dense"), "E3": ("structure", "hybrid"), "E4": ("structure", "hybrid+rerank")}
METRICS = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
COMPARISONS = (("E2", "E1"), ("E3", "E2"), ("E4", "E3"), ("E4", "E1"))  # (new, base): delta = new - base


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()


async def admin(base: str, sql: str) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    eng = create_async_engine(base.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with eng.connect() as c:
        await c.execute(text(sql))
    await eng.dispose()


async def run(base_url: str) -> dict:
    from alembic.config import Config
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from alembic import command
    from app.ingest.chunking import MAX_TOKENS, OVERLAP_MAX
    from app.ingest.pdf import parse_pdf
    from app.ingest.service import create_notebook, get_or_create_user, ingest_pdf
    from app.models import Chunk
    from app.retrieval import CANDIDATES, MODES, RERANK_MODEL, RRF_K, retrieve

    gold = json.loads((Q / "gold.json").read_text(encoding="utf-8"))
    pdfs = {}
    for pid, meta in gold["papers"].items():
        p = Q / "pdfs" / f"{pid}.pdf"
        if not p.exists():
            sys.exit(f"missing {p}; run: python scripts/build_qasper_gold.py")
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["pdf_sha256"] or \
                hashlib.sha256(parse_pdf(data).snapshot.encode("utf-8")).hexdigest() != meta["snapshot_sha256"]:
            sys.exit(f"{pid}: PDF or its parsed snapshot differs from gold.json (arXiv updated the file or the parser changed); "
                     "gold offsets would be wrong. Re-run build_qasper_gold.py and commit the new gold.")
        pdfs[pid] = data

    name = "ct_eval_" + uuid.uuid4().hex[:8]
    await admin(base_url, f'CREATE DATABASE "{name}"')
    url = base_url.rsplit("/", 1)[0] + "/" + name
    os.environ["DATABASE_URL"] = url
    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "alembic"))
        await asyncio.to_thread(command.upgrade, cfg, "head")
        eng = create_async_engine(url, poolclass=NullPool)
        sm = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)

        source_of, ingest_s = {}, 0.0
        async with sm() as s:
            user = await get_or_create_user(s, "eval-user")
            nb = await create_notebook(s, user.id, "qasper-eval")
            await s.commit()
            uid, nid = user.id, nb.id
        for pid, data in pdfs.items():
            t0 = time.perf_counter()
            async with sm() as s:
                src, _ = await ingest_pdf(s, uid, nid, data, f"{pid}.pdf", ("fixed", "structure"))
                await s.commit()
                source_of[pid] = src.id
            ingest_s += time.perf_counter() - t0
        async with sm() as s:
            rows = (await s.execute(select(Chunk.source_id, Chunk.chunker, Chunk.start_char, Chunk.end_char))).all()
        n_chunks = {k: sum(1 for r in rows if r.chunker == k) for k in ("fixed", "structure")}

        questions = gold["questions"]
        golds = {q["id"]: [(g["start"], g["end"]) for g in q["gold"]] for q in questions}
        ceiling = {}  # share of gold spans that ANY chunk of the chunker covers: the best recall that chunker can reach
        for ck in ("fixed", "structure"):
            hit = tot = 0
            for q in questions:
                spans = [(r.start_char, r.end_char) for r in rows if r.chunker == ck and r.source_id == source_of[q["paper"]]]
                cov = frozenset().union(*(covers(c, golds[q["id"]]) for c in spans)) if spans else frozenset()
                hit, tot = hit + len(cov), tot + len(golds[q["id"]])
            ceiling[ck] = hit / tot

        async with sm() as s:  # warm-up: load the embedding and rerank models so latency excludes model loading
            for m in MODES:
                await retrieve(s, uid, nid, "warm up query", m)

        per_q: dict[str, dict] = {q["id"]: {} for q in questions}
        lat: dict[str, dict[str, list[float]]] = {}
        async with sm() as s:
            for exp, (chunker, mode) in EXPERIMENTS.items():
                lat[exp] = {}
                for q in questions:
                    t0 = time.perf_counter()
                    res = await retrieve(s, uid, nid, q["question"], mode, chunker, k=10)
                    total = (time.perf_counter() - t0) * 1000
                    lat[exp].setdefault("total", []).append(total)
                    for stage, ms in res.timings_ms.items():
                        lat[exp].setdefault(stage, []).append(ms)
                    g = golds[q["id"]]
                    ranked = [covers((h.start_char, h.end_char), g) if h.source_id == source_of[q["paper"]] else frozenset()
                              for h in res.hits]
                    per_q[q["id"]][exp] = {"recall@5": recall_covered(ranked, len(g), 5), "recall@10": recall_covered(ranked, len(g), 10),
                                           "mrr@10": mrr_at(ranked, 10), "ndcg@10": ndcg_at(ranked, len(g), 10)}
        await eng.dispose()
    finally:
        await admin(base_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    exps = {}
    for exp, (chunker, mode) in EXPERIMENTS.items():
        exps[exp] = {"chunker": chunker, "mode": mode,
                     "metrics": {m: mean_ci([per_q[q["id"]][exp][m] for q in questions]) for m in METRICS},
                     "latency_ms": {st: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)} for st, v in lat[exp].items()},
                     "provider_cost_usd": 0.0}
    deltas = {f"{new}-{base}": {m: paired_delta_ci([per_q[q["id"]][base][m] for q in questions], [per_q[q["id"]][new][m] for q in questions])
                                for m in METRICS} for new, base in COMPARISONS}
    import torch
    PERQ.write_text(json.dumps(per_q, indent=1) + "\n", encoding="utf-8", newline="\n")
    return {
        "meta": {
            "data": "QASPER dev v0.3 subset; gold = human evidence paragraphs aligned to PDF snapshots (see evals/qasper/gold.json)",
            "setting": "all papers in one notebook (multi-document); one run; no hyperparameter tuned on these questions",
            "gold_sha256": hashlib.sha256((Q / "gold.json").read_bytes()).hexdigest(),
            "papers": len(gold["papers"]), "questions": len(questions), "gold_spans": sum(len(g) for g in golds.values()),
            "alignment": gold["meta"]["counts"], "chunks": n_chunks, "recall_ceiling_by_chunker": ceiling,
            "params": {"candidates": CANDIDATES, "rrf_k": RRF_K, "chunk_max_tokens": MAX_TOKENS, "overlap_max_tokens": OVERLAP_MAX,
                       "embedding_model": "BAAI/bge-small-en-v1.5", "rerank_model": RERANK_MODEL,
                       "fixed_chunker": "350 words, overlap 50"},
            "ingest_seconds_total": round(ingest_s, 1),
            "cost_note": "inference is local (CPU/GPU), provider_cost_usd is 0.0 by construction; compute cost is the latency below",
            "latency_note": "wall ms per query on the machine below, warm models, single run: not portable",
            "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count(),
                        "torch": torch.__version__, "cuda": torch.cuda.is_available()},
            "git": git("rev-parse", "HEAD"), "git_dirty_tracked_files": bool(git("status", "--porcelain", "--untracked-files=no")),
        },
        "experiments": exps, "paired_deltas": deltas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.getenv("EVAL_DATABASE_URL") or os.getenv("TEST_DATABASE_URL"))
    a = ap.parse_args()
    if not a.database_url:
        sys.exit("need --database-url or EVAL_DATABASE_URL / TEST_DATABASE_URL (Postgres with pgvector)")
    res = asyncio.run(run(a.database_url))
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for exp, e in res["experiments"].items():
        m = e["metrics"]
        print(f"{exp} {e['chunker']:<9}{e['mode']:<14}" + "  ".join(f"{k} {v['mean']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for k, v in m.items())
              + f"  p50 {e['latency_ms']['total']['p50']:.0f} ms")
    for k, d in res["paired_deltas"].items():
        r = d["recall@10"]
        print(f"{k}: recall@10 {r['delta']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {'significant' if r['excludes_zero'] else 'not distinguishable'}")


if __name__ == "__main__":
    main()

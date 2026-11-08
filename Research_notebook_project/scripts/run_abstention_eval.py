"""E6a: retrieval-score abstention for chat, calibrated on dev, reported on test. Local compute only, no LLM, $0.

Question: can the top cross-encoder score (hybrid+rerank, the chat retriever) tell "this notebook can answer" from
"it cannot", before any model call?

Setup (all fixed before looking at any score):
- Papers from evals/qasper/gold.json are assigned by SHA-256 of the paper id: 1 in 3 is OUT of the notebook, the rest
  are IN. Independently, papers are split 50/50 into dev and test, so no paper's questions are in both splits.
- answerable    = gold questions about an IN paper (human evidence exists in the notebook).
- unanswerable  = gold questions about an OUT paper (the notebook holds other NLP papers: realistic near-misses).
- unanswerable_in_paper = QASPER questions about an IN paper that every annotator marked unanswerable (needs
  evals/qasper/raw/qasper-dev-v0.3.json from build_qasper_gold.py; few of them, reported separately).
- Policy: answer when top score >= t. t = the largest value keeping >= 90% of DEV answerable questions (fixed target).
  Reported on TEST only: false-abstention rate on answerable, abstention rate on unanswerable, with Wilson CIs; AUROC.
Caveat: "answerable" means the evidence is in the notebook, not that retrieval found it or a model would answer well.

Usage: python scripts/run_abstention_eval.py [--database-url postgresql+asyncpg://user:pw@host:5433/db]
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import auroc, threshold_for_retention, wilson  # noqa: E402
from run_retrieval_eval import admin, git  # noqa: E402

Q = ROOT / "evals" / "qasper"
OUT = ROOT / "evals" / "abstention_results.json"
TARGET_RETENTION = 0.90


def bucket(pid: str, salt: str, n: int) -> int:
    return int(hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest(), 16) % n


def is_out(pid: str) -> bool:
    return bucket(pid, "notebook", 3) == 0


def split(pid: str) -> str:
    return "dev" if bucket(pid, "split", 2) == 0 else "test"


def rates(scores: dict[str, list[float]], t: float) -> dict:
    a, u, ui = scores["answerable"], scores["unanswerable"], scores["unanswerable_in_paper"]
    fa, ab, abi = sum(s < t for s in a), sum(s < t for s in u), sum(s < t for s in ui)
    return {"n_answerable": len(a), "n_unanswerable": len(u), "n_unanswerable_in_paper": len(ui),
            "false_abstention_rate": fa / len(a), "false_abstention_wilson": wilson(fa, len(a)),
            "unanswerable_abstention_rate": ab / len(u), "unanswerable_abstention_wilson": wilson(ab, len(u)),
            "unanswerable_in_paper_abstention": [abi, len(ui)],
            "auroc_answerable_vs_unanswerable": auroc(a, u)}


async def run(base_url: str) -> dict:
    from alembic.config import Config
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from alembic import command
    from app.ingest.service import create_notebook, get_or_create_user, ingest_pdf
    from app.retrieval import RERANK_MODEL, retrieve

    gold = json.loads((Q / "gold.json").read_text(encoding="utf-8"))
    papers_in = sorted(p for p in gold["papers"] if not is_out(p))
    raw = Q / "raw" / "qasper-dev-v0.3.json"
    unans_in: list[dict] = []
    if raw.exists():
        dev = json.loads(raw.read_text(encoding="utf-8"))
        for pid in papers_in:
            for qa in dev[pid]["qas"]:
                if qa["answers"] and all(a["answer"]["unanswerable"] for a in qa["answers"]):
                    unans_in.append({"id": qa["question_id"], "paper": pid, "question": qa["question"]})

    name = "ct_abst_" + uuid.uuid4().hex[:8]
    await admin(base_url, f'CREATE DATABASE "{name}"')
    url = base_url.rsplit("/", 1)[0] + "/" + name
    os.environ["DATABASE_URL"] = url
    rows = []
    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "alembic"))
        await asyncio.to_thread(command.upgrade, cfg, "head")
        eng = create_async_engine(url, poolclass=NullPool)
        sm = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
        async with sm() as s:
            user = await get_or_create_user(s, "eval-user")
            nb = await create_notebook(s, user.id, "abstention-eval")
            await s.commit()
            uid, nid = user.id, nb.id
        for pid in papers_in:
            data = (Q / "pdfs" / f"{pid}.pdf").read_bytes()
            if hashlib.sha256(data).hexdigest() != gold["papers"][pid]["pdf_sha256"]:
                sys.exit(f"{pid}: PDF differs from gold.json; run build_qasper_gold.py")
            async with sm() as s:
                await ingest_pdf(s, uid, nid, data, f"{pid}.pdf")
                await s.commit()
        items = [{**q, "kind": "unanswerable" if is_out(q["paper"]) else "answerable"} for q in gold["questions"]]
        items += [{**q, "kind": "unanswerable_in_paper"} for q in unans_in]
        async with sm() as s:
            for q in items:
                hits = (await retrieve(s, uid, nid, q["question"], "hybrid+rerank", k=1)).hits
                rows.append({"id": q["id"], "paper": q["paper"], "split": split(q["paper"]), "kind": q["kind"],
                             "top_score": hits[0].score if hits else float("-inf")})
        await eng.dispose()
    finally:
        await admin(base_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    by = {sp: {k: [r["top_score"] for r in rows if r["split"] == sp and r["kind"] == k]
               for k in ("answerable", "unanswerable", "unanswerable_in_paper")} for sp in ("dev", "test")}
    t = threshold_for_retention(by["dev"]["answerable"], TARGET_RETENTION)
    return {
        "meta": {"experiment": "E6a retrieval-score abstention (no LLM)", "gold_sha256": hashlib.sha256((Q / "gold.json").read_bytes()).hexdigest(),
                 "papers_in_notebook": len(papers_in), "papers_out": len(gold["papers"]) - len(papers_in),
                 "retriever": f"hybrid+rerank top-1 score ({RERANK_MODEL} logit)", "policy": f"answer iff top score >= t; t keeps >= {TARGET_RETENTION:.0%} of dev answerable",
                 "unanswerable_in_paper_source": "QASPER all-annotators-unanswerable" if raw.exists() else "raw QASPER json missing: not measured",
                 "git": git("rev-parse", "HEAD"), "git_dirty_tracked_files": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "threshold": t, "dev": rates(by["dev"], t), "test": rates(by["test"], t), "per_question": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.getenv("EVAL_DATABASE_URL") or os.getenv("TEST_DATABASE_URL"))
    a = ap.parse_args()
    if not a.database_url:
        sys.exit("need --database-url or EVAL_DATABASE_URL / TEST_DATABASE_URL (Postgres with pgvector)")
    res = asyncio.run(run(a.database_url))
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"threshold {res['threshold']:.3f}")
    for sp in ("dev", "test"):
        r = res[sp]
        print(f"{sp}: false abstention {r['false_abstention_rate']:.3f} {r['false_abstention_wilson']} (n={r['n_answerable']}); "
              f"abstains on unanswerable {r['unanswerable_abstention_rate']:.3f} {r['unanswerable_abstention_wilson']} (n={r['n_unanswerable']}); "
              f"in-paper unanswerable {r['unanswerable_in_paper_abstention']}; AUROC {r['auroc_answerable_vs_unanswerable']:.3f}")


if __name__ == "__main__":
    main()

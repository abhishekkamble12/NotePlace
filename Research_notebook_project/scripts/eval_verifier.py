# scripts/eval_verifier.py
"""
CiteGuard Verifier Evaluation Harness (Week 1 / Spec §9.0).
Measures precision, recall (catch rate), and false-drop rate on the benchmark test set.
Can run against the database table verifier_testset or directly against evals/verifier_testset.csv.
DATA KIND: the CSV is SYNTHETIC (hand-written fixtures), not genuine fetched pages or real drafts. Every report
and run log is labeled data_kind="synthetic_fixture". Snapshot = the row's `context` passage (the stand-in source
document); the citation is resolved to {start, end} offsets in it the same way the pipeline does (locate_quote),
and verify_claim is then called by reference, so Stage 1 (offsets, fact-token check) is exercised, not bypassed.
Rows whose quote cannot be located are counted as citation failures, never skipped.
WARNING: this script calls a live provider (Groq) through call_llm. Results on the current
175-pair set are NOT credible evidence: the set is hand-written, easy, and the prompt was
tuned on it (see AUDIT.md). No baseline or CI gate exists.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz import fuzz

from app.core.verifier import VERIFIER_SYSTEM_PROMPT, locate_quote, verify_claim
from app.models import Job, VerifierTestSet

CSV_PATH = Path(__file__).resolve().parents[1] / "evals" / "verifier_testset.csv"
RUNS_DIR = CSV_PATH.parent / "eval_runs"


async def load_test_items(csv_only: bool = False, limit: int | None = None) -> list[dict]:
    items = []
    # If not forced to csv_only, attempt loading from DB first
    if not csv_only:
        try:
            from sqlalchemy import select

            from app.db import AsyncSessionLocal, init_db
            await init_db()
            async with AsyncSessionLocal() as session:
                res = await session.execute(select(VerifierTestSet))
                db_items = res.scalars().all()
                if db_items:
                    items = [
                        {
                            "id": item.id,
                            "quote": item.quote,
                            "claim": item.claim,
                            "context": item.context,
                            "perturbation_type": item.perturbation_type,
                            "label": item.label,
                            "source": item.source,
                        }
                        for item in db_items
                    ]
        except Exception:
            pass

    if not items and CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append({
                    "id": int(row.get("id", len(items) + 1)),
                    "quote": row["quote"],
                    "claim": row["claim"],
                    "context": row.get("context", ""),
                    "perturbation_type": row.get("perturbation_type", "original"),
                    "label": row["label"],
                    "source": row.get("source", ""),
                })

    if limit is not None:
        items = items[:limit]

    return items


async def run_evaluation(csv_only: bool = False, limit: int | None = None, model: str = "qwen/qwen3.8-27b", baseline: str = ""):
    items = await load_test_items(csv_only=csv_only, limit=limit)

    if not items:
        print("No benchmark items found. Run 'python scripts/generate_testset.py' first.")
        return

    print("\n" + "=" * 65)
    print(f"  CITEGUARD VERIFIER BENCHMARK EVALUATION ({len(items)} samples)")
    print(f"  Model: {model}")
    print("=" * 65 + "\n")

    # Mock DB session for evaluating without requiring active job table write
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.get.return_value = Job(id=999, max_tokens=1_000_000)

    print("  DATA: SYNTHETIC fixtures (hand-written); not genuine pages or real drafts\n")
    rows = []  # (item, predicted verdict, match score)
    unresolved = 0
    for i, item in enumerate(items, 1):
        if baseline == "fuzzy":
            rows.append((item, fuzzy_baseline(item), 0.0))
            continue
        loc = locate_quote(item["quote"], item["context"])
        if loc is None:  # citation failure: counts as a dropped (unsupported) prediction
            unresolved += 1
        res = await verify_claim(
            db=mock_db,
            job_id=999,
            claim_text=item["claim"],
            quote=item["quote"],
            snapshot=item["context"],
            verifier_model=model,
            start=loc[0] if loc else 0,
            end=loc[1] if loc else 0,  # (0, 0) is an invalid ref -> dropped as invalid_citation_ref
        )
        rows.append((item, res.verdict, res.match_score))
        if res.verdict != item["label"]:
            print(f"[{i:>3}/{len(items)}] [{item['perturbation_type']:<15}] Label: {item['label']:<11} Pred: {res.verdict:<11} {res.reason}")

    # Split by quote so all perturbations of a quote share a split (no leakage); heldout is for the final number only.
    splits = {"dev": [r for r in rows if not is_heldout(r[0]["quote"])],
              "heldout": [r for r in rows if is_heldout(r[0]["quote"])],
              "all": rows}
    summary = {name: report(name, rs) for name, rs in splits.items() if rs}

    RUNS_DIR.mkdir(exist_ok=True)
    print(f"\nCitation failures (quote not locatable in snapshot): {unresolved}/{len(rows)}")
    log = {"baseline": baseline or None, "data_kind": "synthetic_fixture", "unresolved_citations": unresolved, "model": model, "n": len(rows), "summary": summary,
           "prompt_sha": hashlib.sha1(VERIFIER_SYSTEM_PROMPT.encode()).hexdigest()[:8]}
    (RUNS_DIR / f"{baseline or log['prompt_sha']}_{model.replace('/', '_')}.json").write_text(json.dumps(log, indent=2))


def fuzzy_baseline(item: dict, threshold: float = 95.0) -> str:
    """No-LLM baseline: claim 'supports' if it fuzzy-matches the cited span. Needs no key; shows what Stage 2 must beat."""
    return "supports" if fuzz.partial_ratio(item["claim"].lower(), item["quote"].lower()) >= threshold else "unsupported"


def is_heldout(quote: str) -> bool:
    return int(hashlib.sha1(quote.encode()).hexdigest(), 16) % 10 < 3  # ~30%


def bootstrap_ci(pairs: list[tuple[bool, bool]], n: int = 1000) -> tuple[float, float]:
    """95% CI of recall over (is_unsupported, predicted_unsupported) pairs."""
    pos = [p for a, p in pairs if a]
    if not pos:
        return 0.0, 0.0
    rng = random.Random(0)
    stats = sorted(sum(rng.choices(pos, k=len(pos))) / len(pos) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def report(name: str, rs: list) -> dict:
    labels = ["supports", "partial", "unsupported"]
    conf = {a: {b: 0 for b in labels} for a in labels}
    by_type = defaultdict(lambda: [0, 0])  # perturbation -> [caught, total] (unsupported only)
    for item, pred, _ in rs:
        conf[item["label"]][pred] += 1
        if item["label"] == "unsupported":
            by_type[item["perturbation_type"]][1] += 1
            by_type[item["perturbation_type"]][0] += pred == "unsupported"
    tp, fn = conf["unsupported"]["unsupported"], conf["unsupported"]["supports"] + conf["unsupported"]["partial"]
    fp = conf["supports"]["unsupported"] + conf["partial"]["unsupported"]
    neg = sum(sum(conf[a].values()) for a in ("supports", "partial"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm = fp / neg if neg else 0.0
    kept = conf["supports"]["supports"] + conf["partial"]["supports"] + conf["unsupported"]["supports"]
    kept_precision = conf["supports"]["supports"] / kept if kept else 0.0
    retention = conf["supports"]["supports"] / sum(conf["supports"].values()) if sum(conf["supports"].values()) else 0.0
    lo, hi = bootstrap_ci([(it["label"] == "unsupported", pred == "unsupported") for it, pred, _ in rs])
    print(f"\n=== {name.upper()} (n={len(rs)}) ===")
    print(f"recall(unsupported)={recall:.1%} [95% CI {lo:.1%}-{hi:.1%}]  precision={precision:.1%}  false-alarm={false_alarm:.1%}")
    print("baseline always-unsupported: recall=100.0% false-alarm=100.0%  (a useful verifier must beat the false-alarm side)")
    print(f"kept-claim precision={kept_precision:.1%} (of {kept} kept)  retention of true-supported claims={retention:.1%}")
    print(f"{'actual/pred':<14}" + "".join(f"{b:>13}" for b in labels))
    for a in labels:
        print(f"{a:<14}" + "".join(f"{conf[a][b]:>13}" for b in labels))
    for t, (c, n) in sorted(by_type.items()):
        print(f"  {t:<15} caught {c}/{n} ({c / n:.0%})")
    return {"recall": recall, "precision": precision, "kept_precision": kept_precision, "retention": retention, "false_alarm": false_alarm, "ci": [lo, hi], "n": len(rs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CiteGuard verifier on test set.")
    parser.add_argument("--csv-only", action="store_true", help="Load directly from CSV without DB")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to evaluate")
    parser.add_argument("--baseline", choices=["fuzzy"], default="", help="Run a no-LLM baseline instead of the verifier")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b", help="Model to evaluate")
    args = parser.parse_args()

    asyncio.run(run_evaluation(csv_only=args.csv_only, limit=args.limit, model=args.model, baseline=args.baseline))

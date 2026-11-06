"""Offline (no API key, no LLM) diagnostic on evals/hard_cases.csv. DATA KIND: synthetic, single annotator.
Arms: stage1 (deterministic fact-token check; cannot emit 'partial', so it predicts supports|unsupported),
fuzzy-N (claim-vs-span partial_ratio >= N, thresholds swept, none chosen). Writes evals/eval_runs/hard_cases_offline.json.
With --stage2 also runs the full verifier (Stage 1 + live Stage 2 LLM, ~24 provider calls, cache disabled) as arm `full`.
Usage: python scripts/eval_hard_cases.py [--stage2]"""
import argparse
import asyncio
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rapidfuzz import fuzz  # noqa: E402

from app.core.verifier import locate_quote, missing_fact_tokens  # noqa: E402

LABELS = ["supports", "partial", "unsupported"]
CSV = ROOT / "evals" / "hard_cases.csv"


def stage1(r: dict) -> str:
    loc = locate_quote(r["quote"], r["context"])
    if loc is None:
        return "unsupported"
    return "unsupported" if missing_fact_tokens(r["claim"], r["context"][loc[0]:loc[1]]) else "supports"


def fuzzy(r: dict, t: float) -> str:
    return "supports" if fuzz.partial_ratio(r["claim"].lower(), r["context"].lower()) >= t else "unsupported"


def metrics(gold: list[str], pred: list[str]) -> dict:
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    for g, p in zip(gold, pred, strict=True):
        conf[g][p] += 1
    per, n = {}, len(gold)
    for c in LABELS:
        tp, fp = conf[c][c], sum(conf[g][c] for g in LABELS) - conf[c][c]
        fn = sum(conf[c].values()) - tp
        p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        per[c] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}
    return {"n": n, "accuracy": sum(conf[c][c] for c in LABELS) / n, "per_class": per,
            "macro_f1": sum(per[c]["f1"] for c in LABELS) / 3,
            "weighted_f1": sum(per[c]["f1"] * per[c]["support"] for c in LABELS) / n, "confusion_rowgold_colpred": conf}


async def full_verifier(rows: list[dict]) -> tuple[list[str], dict]:
    """Product verifier per row (citation located in the context, then verify_claim with live Stage 2) under one real Job,
    so every provider call is budget-checked and ledgered. Needs DATABASE_URL. Returns (verdicts, ledger usage of that job)."""
    from app.core.accounting import ledger_summary
    from app.core.verifier import verify_claim
    from app.db import AsyncSessionLocal, init_db
    from app.models import Job

    await init_db()
    async with AsyncSessionLocal() as db:
        job = Job(question="eval: hard_cases stage2", max_tokens=1_000_000, max_minutes=60, mode="shadow")
        db.add(job)
        await db.commit()
        out = []
        for r in rows:
            loc = locate_quote(r["quote"], r["context"])
            res = await verify_claim(db=db, job_id=job.id, claim_text=r["claim"], quote=r["quote"], snapshot=r["context"],
                                     start=loc[0] if loc else 0, end=loc[1] if loc else 0)
            out.append(res.verdict)
        return out, {"job_id": job.id, **await ledger_summary(db, "job", job.id)}


def main(stage2: bool = False) -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf8")))
    gold = [r["label"] for r in rows]
    arms = {"stage1": [stage1(r) for r in rows], **{f"fuzzy-{t}": [fuzzy(r, t) for r in rows] for t in (60, 70, 80, 90, 95)}}
    if stage2:
        from run_eval_jobs import load_env

        from app.core import llm

        load_env()
        llm.CACHE_DIR = Path(tempfile.mkdtemp(prefix="citeguard_nocache_"))  # read at import, so patch the module: no replay
        arms["full"], usage = asyncio.run(full_verifier(rows))
    out = {"data_kind": "synthetic_single_annotator", "dataset_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "gold_counts": dict(Counter(gold)), "arms": {}}
    if stage2:
        out["stage2_usage"] = usage
    for name, pred in arms.items():
        cat = defaultdict(lambda: [0, 0])
        for r, p in zip(rows, pred, strict=True):
            cat[r["perturbation_type"]][1] += 1
            cat[r["perturbation_type"]][0] += p == r["label"]
        out["arms"][name] = {**metrics(gold, pred), "by_category_correct_total": dict(cat),
                             "errors": [{"id": r["id"], "cat": r["perturbation_type"], "gold": r["label"], "pred": p}
                                        for r, p in zip(rows, pred, strict=True) if p != r["label"]]}
        print(f"{name:<9} acc={out['arms'][name]['accuracy']:.3f} macroF1={out['arms'][name]['macro_f1']:.3f}")
    (ROOT / "evals" / "eval_runs" / f"hard_cases_{'stage2' if stage2 else 'offline'}.json").write_text(json.dumps(out, indent=1))
    if stage2:  # predictions file for scripts/eval_report.py
        with (ROOT / "evals" / "eval_runs" / "hard_cases_full_predictions.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["id", "gold", "pred"])
            w.writerows([r["id"], g, p] for r, g, p in zip(rows, gold, arms["full"], strict=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", action="store_true", help="also run the live LLM verifier (costs ~24 provider calls)")
    main(ap.parse_args().stage2)

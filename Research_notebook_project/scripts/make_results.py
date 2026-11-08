"""`make eval`: recompute every README number from committed data, write evals/results.json, render the README table.
No network, no LLM, deterministic (bootstrap seed 0). `--check` changes nothing and exits 1 if either file is stale,
so a hand-edited README number fails CI.

Inputs (all committed): frozen label sheets evals/real/{dev,test}_label_sheet.csv, run exports evals/real/*_q*.json
(their shadow verdicts are the predictions), evals/hard_cases.csv, evals/eval_runs/hard_cases_full_predictions.csv.
Usage: python scripts/make_results.py [--check]"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hard_cases import CSV as HARD_CSV  # noqa: E402
from eval_hard_cases import fuzzy, metrics, stage1  # noqa: E402
from eval_metrics import LABELS, bootstrap_ci, confusion, counts, wilson  # noqa: E402
from summarize_real_runs import summarize  # noqa: E402

REAL, RUNS = ROOT / "evals" / "real", ROOT / "evals" / "eval_runs"
OUT, README = ROOT / "evals" / "results.json", ROOT / "README.md"
RETR = ROOT / "evals" / "retrieval_results.json"
ABST = ROOT / "evals" / "abstention_results.json"
START, END = "<!-- results:start -->", "<!-- results:end -->"
LABEL_PROVENANCE = "LLM-drafted, author-reviewed, single annotator (no kappa yet)"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def real_predictions() -> list[dict]:
    """Join human labels to the pipeline's shadow verdict by (job, claim_id). Fails if a frozen sheet changed."""
    verdict = {}
    for f in REAL.glob("*_q*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        verdict |= {(d["qid"], str(c["id"])): c["shadow_verdict"] for c in d["claims"]}
    rows = []
    for split in ("dev", "test"):
        sheet = REAL / f"{split}_label_sheet.csv"
        frozen = json.loads(sheet.with_suffix(".freeze.json").read_text())["sha256"]
        if sha(sheet) != frozen:
            raise SystemExit(f"{sheet.name} changed since it was frozen; re-freeze deliberately or revert")
        for r in csv.DictReader(sheet.open(encoding="utf-8")):
            pred = verdict.get((r["job"], r["claim_id"]))
            if r["label"] in LABELS and pred in LABELS:  # skips and verdict-less claims are excluded, and counted below
                rows.append({"id": f'{r["job"]}:{r["claim_id"]}', "split": split, "gold": r["label"], "pred": pred})
    return rows


def verifier_block(rows: list[dict]) -> dict:
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    m = counts(gold, pred)
    return {"metrics": m, "ci95": {**bootstrap_ci(gold, pred),
                                   "far_wilson": wilson(m["false_accepts"], m["gold_not_supports"]),
                                   "precision_wilson": wilson(m["tp"], m["kept"]),
                                   "retention_wilson": wilson(m["tp"], m["gold_supports"])},
            "confusion_rowgold_colpred": confusion(gold, pred)}


def build() -> dict:
    preds = real_predictions()
    hard = list(csv.DictReader(HARD_CSV.open(encoding="utf8")))
    hgold = [r["label"] for r in hard]
    full = {r["id"]: r["pred"] for r in csv.DictReader((RUNS / "hard_cases_full_predictions.csv").open(encoding="utf8"))}
    n_sheet = sum(1 for s in ("dev", "test") for _ in csv.DictReader((REAL / f"{s}_label_sheet.csv").open(encoding="utf-8")))
    return {
        "real_claims": {
            "data": LABEL_PROVENANCE, "predictions": "shadow verdicts recorded at run time (older verifier commits; see RESULTS.md)",
            "sheet_sha256": {s: sha(REAL / f"{s}_label_sheet.csv") for s in ("dev", "test")},
            "rows_in_sheets": n_sheet, "rows_scored": len(preds),
            "all": verifier_block(preds), "test_heldout": verifier_block([r for r in preds if r["split"] == "test"]),
        },
        "hard_cases": {
            "data": "synthetic, author-labeled, diagnostic only", "dataset_sha256": sha(HARD_CSV), "n": len(hard),
            "accuracy": {"full_recorded_live_run": metrics(hgold, [full[r["id"]] for r in hard])["accuracy"],
                         "stage1_current_code": metrics(hgold, [stage1(r) for r in hard])["accuracy"],
                         "fuzzy90_baseline": metrics(hgold, [fuzzy(r, 90) for r in hard])["accuracy"]},
        },
        "pipeline_runs": {"data": "10 live questions, Groq + Tavily; cost modeled from list prices, not billed", **summarize()},
        # recorded by scripts/run_retrieval_eval.py (needs Postgres), so copied here, not recomputed
        "retrieval": {"file": "evals/retrieval_results.json", "file_sha256": sha(RETR), **json.loads(RETR.read_text(encoding="utf-8"))},
        "abstention": {"file": "evals/abstention_results.json", "file_sha256": sha(ABST),
                       **{k: v for k, v in json.loads(ABST.read_text(encoding="utf-8")).items() if k != "per_question"}},
    }


def ci(v: list[float]) -> str:
    return f"{v[0]:.3f}-{v[1]:.3f}"


def table(r: dict) -> str:
    rc, h, p = r["real_claims"], r["hard_cases"], r["pipeline_runs"]
    rows = []
    for name, b in (("all", rc["all"]), ("test split (held out)", rc["test_heldout"])):
        m, c = b["metrics"], b["ci95"]
        rows.append(f"| Verifier on real claims, {name} (n={m['n']}) | false-acceptance {m['false_accepts']}/{m['gold_not_supports']}"
                    f" = {m['far']:.3f} (Wilson {ci(c['far_wilson'])}); precision {m['precision']:.3f} (Wilson {ci(c['precision_wilson'])});"
                    f" retention {m['retention']:.3f} (Wilson {ci(c['retention_wilson'])}) | {rc['data']} |")
    a = h["accuracy"]
    rows.append(f"| Verifier on hard cases: full (recorded live run) vs Stage 1 (current code) vs fuzzy-90 baseline |"
                f" 3-class accuracy {a['full_recorded_live_run']:.3f} vs {a['stage1_current_code']:.3f} vs {a['fuzzy90_baseline']:.3f}"
                f" | {h['n']} cases, {h['data']} |")
    rows.append(f"| Real pipeline runs | {p['jobs']} jobs, {p['llm_calls']} LLM calls, ${p['usd_modeled']} total, per-call latency"
                f" p50 {p['call_latency_ms_p50'] / 1000:.2f} s / p95 {p['call_latency_ms_p95'] / 1000:.2f} s | {p['data']} |")
    rm, e = r["retrieval"]["meta"], r["retrieval"]["experiments"]
    r10 = {k: e[k]["metrics"]["recall@10"] for k in e}
    d = r["retrieval"]["paired_deltas"]["E4-E3"]["recall@10"]
    rows.append(f"| PDF retrieval recall@10 (E1 fixed+dense / E2 structure+dense / E3 +hybrid / E4 +rerank) |"
                f" {' / '.join(f'{r10[k]['mean']:.3f}' for k in ('E1', 'E2', 'E3', 'E4'))}; rerank gain {d['delta']:+.3f}"
                f" (paired CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}); E4 p50 {e['E4']['latency_ms']['total']['p50'] / 1000:.2f} s on CPU"
                f" | QASPER dev, {rm['questions']} questions over {rm['papers']} papers in one notebook, human evidence spans; recorded run |")
    ab = r["abstention"]["test"]
    rows.append(f"| Chat abstention from retrieval score alone (E6a, test split; gate left off) |"
                f" refuses {ab['false_abstention_rate']:.3f} of answerable (Wilson {ci(ab['false_abstention_wilson'])}) to catch"
                f" {ab['unanswerable_abstention_rate']:.3f} of unanswerable (Wilson {ci(ab['unanswerable_abstention_wilson'])}); AUROC"
                f" {ab['auroc_answerable_vs_unanswerable']:.3f} | QASPER, {ab['n_answerable']} answerable / {ab['n_unanswerable']}"
                f" out-of-notebook questions, threshold set on dev papers; recorded run |")
    return "\n".join([START, "| Measurement | Value | Data |", "|---|---|---|", *rows, END])


def main() -> None:
    res = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    readme = README.read_text(encoding="utf-8")
    i, j = readme.index(START), readme.index(END) + len(END)
    new_readme = readme[:i] + table(json.loads(res)) + readme[j:]
    if "--check" in sys.argv:
        stale = [p.name for p, new in ((OUT, res), (README, new_readme)) if not p.exists() or p.read_text(encoding="utf-8") != new]
        sys.exit(f"stale, run `make eval`: {stale}" if stale else 0)
    OUT.write_text(res, encoding="utf-8", newline="\n")
    README.write_text(new_readme, encoding="utf-8", newline="\n")
    print(table(json.loads(res)))


if __name__ == "__main__":
    main()

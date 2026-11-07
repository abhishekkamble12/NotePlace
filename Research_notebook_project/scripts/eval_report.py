"""Report verifier quality from a predictions CSV (columns: id, gold, pred[, score]). No LLM, no network.

Usage: python scripts/eval_report.py PREDICTIONS.csv [--out evals/eval_runs/report.json]
Gold must come from human labels (see docs/ANNOTATION_GUIDELINES.md); the report records the file SHA-256 and git HEAD.
Prints precision next to retention, bootstrap 95% CIs, the confusion matrix and, if `score` exists, the precision-vs-coverage sweep.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import LABELS, bootstrap_ci, confusion, counts, precision_coverage_sweep, wilson  # noqa: E402


def build(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    bad = [r["id"] for r in rows if r["gold"] not in LABELS or r["pred"] not in LABELS]
    if bad:
        raise SystemExit(f"invalid gold/pred labels in rows: {bad[:10]}")
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    out = {
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "metrics": (m := counts(gold, pred)), "ci95": {**bootstrap_ci(gold, pred), "precision_wilson": wilson(m["tp"], m["kept"]), "retention_wilson": wilson(m["tp"], m["gold_supports"]), "far_wilson": wilson(m["false_accepts"], m["gold_not_supports"])}, "confusion_rowgold_colpred": confusion(gold, pred),
    }
    if rows and rows[0].get("score"):
        out["precision_vs_coverage"] = precision_coverage_sweep(gold, [float(r["score"]) for r in rows])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rep = build(a.predictions)
    m, ci = rep["metrics"], rep["ci95"]
    print(f"n={m['n']} kept={m['kept']} gold_supports={m['gold_supports']}")
    for k in ("far", "precision", "retention", "f1"):
        print(f"{k:<10} {m[k]:.3f}  95% CI bootstrap [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
              + (f"  Wilson [{ci[k + '_wilson'][0]:.3f}, {ci[k + '_wilson'][1]:.3f}]" if k + "_wilson" in ci else ""))
    print("confusion (rows gold, cols pred):", json.dumps(rep["confusion_rowgold_colpred"]))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

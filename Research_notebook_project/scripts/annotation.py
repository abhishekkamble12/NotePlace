"""Annotation tooling. One file per annotator (never merged in place); gold = separately adjudicated file.
  python scripts/annotation.py validate FILE.csv [--allow-empty-labels]
  python scripts/annotation.py agree A.csv B.csv     # raw agreement, Cohen's kappa, per-class, confusion
Disagreements are listed, never resolved here: adjudication is a human edit with adjudication_status=adjudicated."""
import csv
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohens_kappa import kappa  # noqa: E402

COLUMNS = ["pair_id", "source_document_id", "source_url", "source_date", "claim", "evidence", "expected_label",
           "category", "annotator_id", "annotation_notes", "adjudication_status"]
LABELS = ["supports", "partial", "unsupported"]
CATEGORIES = {"paraphrase", "rounding", "numeric_mismatch", "negation", "hedging", "temporal", "entity", "causal",
              "multi_hop", "partial", "contradiction", "missing_evidence", "overgeneralization", "scope", "ambiguous",
              "outdated", "multi_claim", "adversarial", "direct"}
STATUSES = {"none", "agreed", "pending", "adjudicated"}


def validate(rows: list[dict], allow_empty_labels: bool = False) -> list[str]:
    errs, seen, pairs = [], set(), {}
    if rows and set(COLUMNS) - set(rows[0]):
        return [f"missing columns: {sorted(set(COLUMNS) - set(rows[0]))}"]
    for n, r in enumerate(rows, 2):  # line numbers incl. header
        k = (r["pair_id"], r["annotator_id"])
        if not r["pair_id"].strip():
            errs.append(f"line {n}: empty pair_id")
        if k in seen:
            errs.append(f"line {n}: duplicate (pair_id, annotator_id) {k}")
        seen.add(k)
        for c in ("source_document_id", "claim", "evidence", "annotator_id"):
            if not r[c].strip():
                errs.append(f"line {n}: empty {c}")
        u = urlparse(r["source_url"])
        if u.scheme not in ("http", "https") or not u.netloc:
            errs.append(f"line {n}: invalid source_url {r['source_url']!r}")
        if r["category"] not in CATEGORIES:
            errs.append(f"line {n}: unknown category {r['category']!r}")
        if r["expected_label"] not in LABELS and not (allow_empty_labels and not r["expected_label"]):
            errs.append(f"line {n}: invalid label {r['expected_label']!r}")
        if r["adjudication_status"] not in STATUSES:
            errs.append(f"line {n}: invalid adjudication_status {r['adjudication_status']!r}")
        key = (" ".join(r["claim"].lower().split()), " ".join(r["evidence"].lower().split()), r["annotator_id"])
        if key in pairs and pairs[key] != r["pair_id"]:
            errs.append(f"line {n}: duplicate claim/evidence of pair {pairs[key]}")
        pairs.setdefault(key, r["pair_id"])
    return errs


def agreement(a: dict[str, str], b: dict[str, str]) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    x, y = [a[i] for i in ids], [b[i] for i in ids]
    conf = {p: {q: 0 for q in LABELS} for p in LABELS}
    for p, q in zip(x, y, strict=True):
        conf[p][q] += 1
    per_class = {c: (sum(p == q == c for p, q in zip(x, y, strict=True)) / n if (n := x.count(c) + y.count(c) - sum(p == q == c for p, q in zip(x, y, strict=True))) else None) for c in LABELS}
    return {"n": len(ids), "raw_agreement": sum(p == q for p, q in zip(x, y, strict=True)) / len(ids),
            "cohens_kappa": kappa(x, y), "per_class_jaccard": per_class, "confusion_rowA_colB": conf,
            "disagreements": [i for i in ids if a[i] != b[i]]}


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "validate":
        e = validate(_load(sys.argv[2]), "--allow-empty-labels" in sys.argv)
        print("\n".join(e) or "OK")
        sys.exit(1 if e else 0)
    if cmd == "agree":
        la = {r["pair_id"]: r["expected_label"] for r in _load(sys.argv[2])}
        lb = {r["pair_id"]: r["expected_label"] for r in _load(sys.argv[3])}
        print(Counter(la.values()), Counter(lb.values()))
        import json
        print(json.dumps(agreement(la, lb), indent=1))

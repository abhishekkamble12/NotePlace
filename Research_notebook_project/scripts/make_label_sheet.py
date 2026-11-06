# scripts/make_label_sheet.py
"""Build a blind labeling CSV from evals/real/<split>_*.json. Verifier verdicts are NOT included.
Rows are shuffled with a fixed seed; every 4th row is flagged kappa_subset for the second annotator.
Usage: python scripts/make_label_sheet.py dev|test"""
import csv
import json
import random
import sys
from pathlib import Path

REAL = Path(__file__).resolve().parents[1] / "evals" / "real"


def build(split: str) -> list[dict]:
    rows = []
    for f in sorted(REAL.glob(f"{split}_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        snaps = {s["id"]: s["snapshot"] for s in d["sources"]}
        for c in d["claims"]:
            snap = snaps.get(c["source_id"])
            if snap is None or c["start_char"] is None:
                continue  # Stage-1 failures have no span to label; they are counted in the export, not here
            s, e = c["start_char"], c["end_char"]
            rows.append({
                "job": d["qid"], "claim_id": c["id"], "claim": c["text"], "span": snap[s:e],
                "context": snap[max(0, s - 400):e + 400], "label": "", "annotator": "", "notes": "",
            })
    random.Random(0).shuffle(rows)
    for i, r in enumerate(rows):
        r["kappa_subset"] = int(i % 4 == 0)
    return rows


if __name__ == "__main__":
    split = sys.argv[1]
    rows = build(split)
    out = REAL / f"{split}_label_sheet.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows, {sum(r['kappa_subset'] for r in rows)} in kappa subset")

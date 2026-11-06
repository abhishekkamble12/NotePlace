# scripts/freeze_dataset.py
"""Freeze a labeled CSV: write SHA-256 + row/class counts to evals/real/<name>.freeze.json.
`--check` re-hashes and fails if the file changed. Eval results must record this dataset version.
Usage: python scripts/freeze_dataset.py evals/real/test_labeled.csv [--check]"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
meta = path.with_suffix(".freeze.json")
sha = hashlib.sha256(path.read_bytes()).hexdigest()
if "--check" in sys.argv:
    want = json.loads(meta.read_text())["sha256"]
    sys.exit(0 if sha == want else f"FROZEN FILE CHANGED: {sha} != {want}")
with open(path, encoding="utf-8") as f:
    labels = Counter(r["label"] for r in csv.DictReader(f))
meta.write_text(json.dumps({"file": path.name, "sha256": sha, "class_counts": dict(labels)}, indent=1))
print(f"frozen {path.name} sha256={sha[:12]}… {dict(labels)}")

# scripts/cohens_kappa.py
"""Cohen's kappa between two annotators on the kappa_subset rows. Joins on claim_id; 'skip' rows are excluded and counted.
Usage: python scripts/cohens_kappa.py annotator1.csv annotator2.csv"""
import csv
import sys
from collections import Counter


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["claim_id"]: r["label"].strip() for r in csv.DictReader(f) if r["label"].strip()}


if __name__ == "__main__":
    l1, l2 = load(sys.argv[1]), load(sys.argv[2])
    ids = [i for i in l1 if i in l2 and "skip" not in (l1[i], l2[i])]
    k = kappa([l1[i] for i in ids], [l2[i] for i in ids])
    print(f"n={len(ids)} (excluded {len(set(l1) & set(l2)) - len(ids)} skips) agreement={sum(l1[i] == l2[i] for i in ids) / len(ids):.3f} kappa={k:.3f}")

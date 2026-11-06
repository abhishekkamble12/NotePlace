# scripts/check_source_overlap.py
"""Dev/test leakage check on SOURCES. Compares normalized URLs and snapshot content hashes across the dev_* and
test_* exports in evals/real/. Exit 1 if any source appears in both splits.
Handling policy: the test split is frozen, so overlapping sources are removed from DEV (claims citing them are
excluded from tuning); the test set is never edited. Run before any test evaluation."""
import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

REAL = Path(__file__).resolve().parents[1] / "evals" / "real"
_TRACK = ("utm_", "fbclid", "gclid", "ref")


def norm_url(u: str) -> str:
    p = urlsplit(u.strip())
    host = p.netloc.lower().removeprefix("www.")
    q = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(_TRACK)))
    return f"{host}{p.path.rstrip('/')}" + (f"?{q}" if q else "")


def sources(split: str, root: Path = REAL) -> dict[str, list[str]]:
    """key (normalized url or 'hash:<sha>') -> jobs that cite it"""
    out: dict[str, list[str]] = {}
    for f in sorted(root.glob(f"{split}_q*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sources"]:
            for k in (norm_url(s["url"]), "hash:" + s["content_hash"]):
                out.setdefault(k, []).append(d["qid"])
    return out


def overlaps(root: Path = REAL) -> dict[str, tuple[list[str], list[str]]]:
    dev, test = sources("dev", root), sources("test", root)
    return {k: (dev[k], test[k]) for k in dev.keys() & test.keys()}


if __name__ == "__main__":
    ov = overlaps()
    for k, (a, b) in sorted(ov.items()):
        print(f"OVERLAP {k}: dev jobs {a} / test jobs {b}")
    print(f"{len(ov)} overlapping sources (dev vs test)")
    sys.exit(1 if ov else 0)

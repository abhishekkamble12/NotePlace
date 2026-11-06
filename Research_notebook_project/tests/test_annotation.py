import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from annotation import COLUMNS, agreement, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def row(**kw):
    r = dict(zip(COLUMNS, ["p1", "d1", "https://a.org/x", "2024-01-01", "c", "e", "supports", "direct", "A", "", "none"], strict=True))
    return {**r, **kw}


def test_template_valid_when_labels_empty():
    rows = list(csv.DictReader((ROOT / "evals" / "annotation_template.csv").open(encoding="utf8")))
    assert validate(rows, allow_empty_labels=True) == []
    assert validate(rows)  # empty label is an error for a real file


def test_validator_catches_problems():
    assert validate([row()]) == []
    assert any("duplicate (pair_id" in e for e in validate([row(), row()]))
    assert any("invalid label" in e for e in validate([row(expected_label="yes")]))
    assert any("source_url" in e for e in validate([row(source_url="ftp://x")]))
    assert any("category" in e for e in validate([row(category="zzz")]))
    assert any("duplicate claim/evidence" in e for e in validate([row(), row(pair_id="p2")]))
    assert validate([row(), row(annotator_id="B")]) == []  # same pair, two annotators is legal


def test_agreement_known_values():
    a = {"1": "supports", "2": "partial", "3": "unsupported", "4": "unsupported"}
    b = {"1": "supports", "2": "unsupported", "3": "unsupported", "4": "unsupported"}
    r = agreement(a, b)
    assert r["raw_agreement"] == 0.75 and r["disagreements"] == ["2"]
    assert abs(r["cohens_kappa"] - 5 / 9) < 1e-9  # po=.75, pe=(1*1+2*3)/16=.4375
    assert agreement(a, {"9": "supports"}) == {"n": 0}

import csv
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "evals" / "hard_cases.csv"


def test_hard_cases_wellformed():
    rows = list(csv.DictReader(P.open(encoding="utf8")))
    assert len(rows) >= 20
    assert {r["label"] for r in rows} <= {"supports", "partial", "unsupported"}
    assert {"negation", "rounding", "hedging", "multi_hop", "overstated", "adversarial_snapshot", "partial", "paraphrase"} <= {
        r["perturbation_type"] for r in rows}
    assert all(r["quote"] in r["context"] for r in rows)

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_metrics import bootstrap_ci, confusion, counts, precision_coverage_sweep, recall_at_k, wilson  # noqa: E402
from eval_report import build  # noqa: E402

GOLD = ["supports", "supports", "partial", "unsupported", "unsupported", "supports"]
PRED = ["supports", "unsupported", "supports", "unsupported", "supports", "supports"]


def test_counts_hand_computed():
    c = counts(GOLD, PRED)  # kept=4, tp=2, gold supports=3
    assert (c["kept"], c["tp"], c["gold_supports"]) == (4, 2, 3)
    assert c["precision"] == 0.5 and math.isclose(c["retention"], 2 / 3) and math.isclose(c["f1"], 4 / 7)


def test_all_dropped_precision_undefined_not_one():
    c = counts(GOLD, ["unsupported"] * 6)
    assert math.isnan(c["precision"]) and c["retention"] == 0.0


def test_confusion_totals():
    assert sum(sum(r.values()) for r in confusion(GOLD, PRED).values()) == 6


def test_bootstrap_brackets_point_and_is_deterministic():
    ci = bootstrap_ci(GOLD * 10, PRED * 10, n_boot=500)
    p = counts(GOLD * 10, PRED * 10)["precision"]
    assert ci["precision"][0] <= p <= ci["precision"][1] and ci == bootstrap_ci(GOLD * 10, PRED * 10, n_boot=500)


def test_sweep_monotone_coverage():
    rows = precision_coverage_sweep(GOLD, [0.9, 0.8, 0.7, 0.6, 0.5, 0.95])
    assert [r["kept"] for r in rows] == sorted(r["kept"] for r in rows)[::-1]
    assert rows[0]["retention"] == 1.0 and rows[-1]["kept"] == 1


def test_recall_at_k():
    assert recall_at_k([["a", "b", "c"], ["x"]], [{"a", "c"}, set()], 2) == 0.5


def test_report_rejects_bad_labels(tmp_path):
    p = tmp_path / "p.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows([["id", "gold", "pred"], ["1", "yes", "supports"]])
    try:
        build(p)
    except SystemExit as e:
        assert "invalid" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_wilson_not_degenerate_on_all_correct():
    lo, hi = wilson(5, 5)
    assert hi == 1.0 and 0.56 < lo < 0.57  # textbook value 0.566

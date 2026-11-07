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


def test_false_acceptance_rate_hand_computed():
    # 2 kept non-supports out of 4 gold non-supports; 1 of 3 gold supports dropped
    gold = ["supports", "supports", "supports", "partial", "partial", "unsupported", "unsupported"]
    pred = ["supports", "supports", "unsupported", "supports", "partial", "supports", "unsupported"]
    m = counts(gold, pred)
    assert (m["false_accepts"], m["gold_not_supports"]) == (2, 4)
    assert m["far"] == 0.5
    assert math.isclose(m["frr"], 1 / 3)


def test_far_undefined_without_negatives():
    assert math.isnan(counts(["supports"], ["supports"])["far"])


def test_bootstrap_reports_far():
    gold = ["supports", "unsupported"] * 20
    pred = ["supports", "supports", "supports", "unsupported"] * 10
    ci = bootstrap_ci(gold, pred, n_boot=200)
    assert ci["far"][0] <= 0.5 <= ci["far"][1]


import pytest  # noqa: E402
from eval_metrics import covers, mean_ci, mrr_at, ndcg_at, recall_covered  # noqa: E402


def test_covers_overlap_rule():
    gold = [(100, 200), (1000, 1100)]
    assert covers((90, 210), gold) == {0}                  # chunk contains the gold span
    assert covers((120, 150), gold) == {0}                 # chunk inside the gold span: overlap == shorter side
    assert covers((50, 140), gold) == frozenset()          # 40 < 0.5 * 90
    assert covers((50, 400), gold) == {0}                  # chunk spans the gold span fully
    assert covers((0, 5000), gold) == {0, 1}               # a huge chunk covers both


def test_covers_boundary_is_inclusive_at_half():
    assert covers((150, 400), [(100, 200)]) == {0}         # overlap 50 == 0.5 * min(250, 100)
    assert covers((151, 400), [(100, 200)]) == frozenset()  # overlap 49


def test_rank_metrics_hand_computed():
    e = frozenset()
    ranked = [e, frozenset({0}), e, frozenset({1}), e]     # relevant at ranks 2 and 4, two gold spans
    assert recall_covered(ranked, 2, 1) == 0 and recall_covered(ranked, 2, 2) == 0.5 and recall_covered(ranked, 2, 5) == 1.0
    assert mrr_at(ranked, 10) == 0.5 and mrr_at(ranked, 1) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at(ranked, 2, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at([frozenset({0}), e], 1, 10) == 1.0       # perfect ranking
    assert recall_covered([], 3, 5) == 0.0 and mrr_at([], 10) == 0.0


def test_duplicate_coverage_counts_for_recall_once():
    ranked = [frozenset({0}), frozenset({0})]
    assert recall_covered(ranked, 2, 10) == 0.5


def test_mean_ci_brackets_and_is_deterministic():
    v = [0.0, 1.0] * 20
    r = mean_ci(v)
    assert r["mean"] == 0.5 and r["ci95"][0] < 0.5 < r["ci95"][1] and r == mean_ci(v)


from eval_metrics import paired_delta_ci  # noqa: E402


def test_paired_delta_detects_consistent_gain_and_not_noise():
    a = [0.0, 0.2, 0.4, 0.6] * 25
    gain = paired_delta_ci(a, [x + 0.1 for x in a])
    assert gain["delta"] == pytest.approx(0.1) and gain["excludes_zero"] is True
    noise = paired_delta_ci([0.0, 1.0] * 50, [1.0, 0.0] * 50)   # same mean, opposite per-query outcomes
    assert noise["delta"] == 0 and noise["excludes_zero"] is False


def test_paired_delta_rejects_unpaired_lengths():
    with pytest.raises(ValueError):
        paired_delta_ci([1.0, 2.0], [1.0])

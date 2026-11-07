"""The QASPER gold builder locates evidence in the snapshot; it must not invent offsets or match unrelated text."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qasper_align import align, normalize_with_map  # noqa: E402

EVIDENCE = ("We evaluate the model on three benchmark datasets and report macro averaged F1 scores, "
            "finding that the proposed method outperforms all baselines by a clear margin in every setting.")
PAD = "Unrelated opening text about something else entirely, with numbers 123 and names. "
TAIL = " Closing remarks that continue for a while and do not matter here at all."


def align_text(ev, snap):
    r = align(ev, snap)
    return snap[r["start"]:r["end"]]


def test_pdf_noise_is_tolerated():
    noisy = EVIDENCE.replace("benchmark", "bench-\nmark").replace("proposed", "pro­posed").replace("outperforms", "outperﬁorms")
    r = align(EVIDENCE, PAD + noisy + TAIL)
    assert r and r["score"] >= 90
    assert align_text(EVIDENCE, PAD + EVIDENCE + TAIL).startswith("We evaluate the model")


def test_offsets_are_real_slices_covering_the_evidence():
    snap = PAD + EVIDENCE + TAIL
    r = align(EVIDENCE, snap)
    assert (r["method"], r["score"]) == ("exact", 100.0)
    assert snap[r["start"]:r["end"]] == EVIDENCE.rstrip(".")  # the match ends at the last alphanumeric character


def test_citation_markers_and_dataset_tokens_are_ignored():
    ev = EVIDENCE.replace("scores,", "scores BIBREF3 , INLINEFORM0 ")
    snap = PAD + EVIDENCE + TAIL
    r = align(ev, snap)
    assert r and snap[r["start"]:r["end"]].startswith("We evaluate")


def test_citation_numbers_in_pdf_text_still_align_fuzzily():
    snap = PAD + EVIDENCE.replace("scores,", "scores [12, 14],").replace("clear margin", "clear margin (Smith et al., 2019)") + TAIL
    r = align(EVIDENCE, snap)
    assert r and r["method"] == "fuzzy" and r["score"] >= 90


def test_unrelated_text_is_not_aligned():
    other = ("The committee met on Tuesday to discuss the budget for the coming year and agreed to postpone "
             "the vote on the new regional infrastructure plan until further consultation could take place.")
    assert align(EVIDENCE, PAD + other + TAIL) is None


def test_short_evidence_is_refused():
    assert align("Table 1 shows results.", PAD + "Table 1 shows results." + TAIL) is None


def test_normalize_map_points_back_to_original_indices():
    text, idx = normalize_with_map("Ab-\nﬁ c")
    assert text == "abfic" and all("".join(ch for ch in "Ab-\nﬁ c"[i]).strip() for i in idx)
    assert idx[0] == 0 and idx[1] == 1

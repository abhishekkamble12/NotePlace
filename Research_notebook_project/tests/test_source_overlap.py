import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_source_overlap import norm_url, overlaps  # noqa: E402


def test_norm_url_collapses_variants():
    a = norm_url("https://www.Example.com/a/b/?utm_source=x&id=1#frag")
    assert a == norm_url("http://example.com/a/b?id=1") == "example.com/a/b?id=1"


def test_overlap_detected_by_url_and_hash(tmp_path):
    def w(name, qid, url, h):
        (tmp_path / name).write_text(json.dumps({"qid": qid, "sources": [{"url": url, "content_hash": h}]}))
    w("dev_q1.json", "q1", "https://a.com/x/", "h1")
    w("test_q2.json", "q2", "http://www.a.com/x", "h2")   # same URL, different hash
    w("dev_q5.json", "q5", "https://b.com/y", "h9")
    w("test_q8.json", "q8", "https://c.com/z", "h9")      # different URL, same content
    assert set(overlaps(tmp_path)) == {"a.com/x", "hash:h9"}

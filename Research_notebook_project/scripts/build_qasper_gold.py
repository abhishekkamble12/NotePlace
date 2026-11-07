"""Build the retrieval gold file evals/qasper/gold.json from QASPER (dev split, v0.3, Allen AI, CC BY 4.0).

QASPER gives questions over NLP papers with human-annotated EVIDENCE PARAGRAPHS (for answerable questions). We
download the papers' arXiv PDFs, parse them with the same PyMuPDF path the product uses, and locate each evidence
paragraph in the snapshot (scripts/qasper_align.py). Evidence that cannot be located with score >= 90 is dropped and
counted (this includes the ~3% of evidence that is a table or figure caption). Offsets are matched, never invented.

Selection is deterministic and not cherry-picked: papers in sorted-id order; the first N whose PDF downloads, parses
and yields at least one question with aligned gold. Nothing here is tuned against any retrieval result.

Network: dataset tgz (11 MB, cached under evals/qasper/raw) and arXiv PDFs (cached under evals/qasper/pdfs, >=3 s apart).
Usage: python scripts/build_qasper_gold.py [--papers 25]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qasper_align import align, normalize_with_map  # noqa: E402

from app.ingest.pdf import PdfError, parse_pdf  # noqa: E402

Q = ROOT / "evals" / "qasper"
DATA_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
UA = "citeguard-eval/0.1 (research reproduction script)"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, dest: Path) -> bytes | None:
    if dest.exists():
        return dest.read_bytes()
    time.sleep(3)  # arXiv asks for no more than one request per 3 seconds
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
            data = r.read()
    except Exception as e:  # network or HTTP error: recorded by the caller as a rejected paper
        print(f"  download failed {url}: {type(e).__name__}")
        return None
    dest.write_bytes(data)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=25)
    n_papers = ap.parse_args().papers
    (Q / "raw").mkdir(parents=True, exist_ok=True)
    (Q / "pdfs").mkdir(parents=True, exist_ok=True)
    tgz = fetch(DATA_URL, Q / "raw" / "qasper-train-dev-v0.3.tgz")
    assert tgz, "could not download QASPER"
    dev_path = Q / "raw" / "qasper-dev-v0.3.json"
    if not dev_path.exists():
        with tarfile.open(Q / "raw" / "qasper-train-dev-v0.3.tgz") as t:
            t.extract("qasper-dev-v0.3.json", Q / "raw", filter="data")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))

    papers, questions, rejected = {}, [], {}
    c = {"evidence_total": 0, "unaligned_dropped": 0, "aligned_exact": 0, "aligned_fuzzy": 0,
         "questions_seen": 0, "questions_unanswerable_or_no_evidence": 0, "questions_no_aligned_gold": 0}
    for pid in sorted(dev):
        if len(papers) >= n_papers:
            break
        pdf = fetch(f"https://arxiv.org/pdf/{pid}", Q / "pdfs" / f"{pid}.pdf")
        if pdf is None:
            rejected[pid] = "download_failed"
            continue
        try:
            doc = parse_pdf(pdf)
        except PdfError as e:
            rejected[pid] = f"pdf_error: {e}"
            continue
        norm = normalize_with_map(doc.snapshot)
        cache: dict[str, dict | None] = {}
        paper_qs = []
        for qa in dev[pid]["qas"]:
            c["questions_seen"] += 1
            evid = []
            for a in qa["answers"]:
                if not a["answer"]["unanswerable"]:
                    evid += a["answer"]["evidence"]
            evid = list(dict.fromkeys(e for e in evid if e.strip()))
            if not evid:
                c["questions_unanswerable_or_no_evidence"] += 1
                continue
            gold = []
            for e in evid:
                c["evidence_total"] += 1
                if e not in cache:
                    cache[e] = align(e, doc.snapshot, norm)
                r = cache[e]
                if r is None:
                    c["unaligned_dropped"] += 1
                else:
                    c["aligned_" + r["method"]] += 1
                    gold.append(r)
            gold = [g for i, g in enumerate(gold) if (g["start"], g["end"]) not in {(h["start"], h["end"]) for h in gold[:i]}]
            if not gold:
                c["questions_no_aligned_gold"] += 1
                continue
            paper_qs.append({"id": qa["question_id"], "paper": pid, "question": qa["question"], "gold": gold,
                             "n_evidence": len(evid)})
        if not paper_qs:
            rejected[pid] = "no_question_with_aligned_gold"
            continue
        papers[pid] = {"title": dev[pid]["title"], "pdf_sha256": sha(pdf), "snapshot_sha256": sha(doc.snapshot.encode("utf-8")),
                       "n_pages": doc.n_pages, "snapshot_chars": len(doc.snapshot)}
        questions += paper_qs
        print(f"{pid}: {len(paper_qs)} questions, {doc.n_pages} pages")

    out = {"meta": {"source": "QASPER dev v0.3 (Allen AI, CC BY 4.0)", "dataset_tgz_sha256": sha(tgz), "selection": "first N sorted ids with usable PDF and >=1 aligned question",
                    "align_min_score": 90, "counts": c, "papers": len(papers), "questions": len(questions), "rejected_papers": rejected,
                    "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()},
           "papers": papers, "questions": questions}
    (Q / "gold.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["meta"], indent=1)[:1500])


if __name__ == "__main__":
    main()

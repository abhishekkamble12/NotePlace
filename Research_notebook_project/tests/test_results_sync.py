"""README numbers are generated: a hand-edited metric, a changed frozen label sheet, or a verifier change that moves
an offline number fails here until `make eval` is re-run and committed."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_results_json_match_committed_data():
    r = subprocess.run([sys.executable, "scripts/make_results.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout

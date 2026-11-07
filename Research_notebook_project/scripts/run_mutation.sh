#!/bin/bash
# Usage (inside python:3.12-slim, see Makefile): bash scripts/run_mutation.sh [target.py] ["test files"]
# Runs mutmut on a throwaway COPY of the repo so pyproject.toml is never edited. Tests that need Postgres/network are excluded.
set -e
TARGET="${1:-app/core/verifier.py}"
TESTS="${2:-tests/test_verifier_phase1.py tests/property/test_verifier_properties.py tests/sec/test_injection_corpus.py tests/test_hard_cases.py tests/test_phase1_followups.py tests/test_verifier_branches.py tests/test_verifier_survivors.py}"
pip install -q "mutmut<4" pytest pytest-asyncio hypothesis sqlalchemy[asyncio] asyncpg pgvector groq pydantic rapidfuzz httpx trafilatura pytest-randomly alembic langgraph mcp 2>&1 | tail -2 || true
export CITEGUARD_EVIDENCE_HMAC_KEY=test-key-not-a-secret
rm -rf /tmp/w && mkdir /tmp/w && cp -r app tests scripts evals alembic alembic.ini pyproject.toml /tmp/w/ && cd /tmp/w
python - "$TARGET" "$TESTS" <<'PY'
import re, sys
target, tests = sys.argv[1], sys.argv[2].split()
s = open("pyproject.toml").read()
s = re.sub(r'source_paths = \[.*?\]', f'source_paths = ["{target}"]', s)
s = re.sub(r'pytest_add_cli_args_test_selection = \[.*?\]', 'pytest_add_cli_args_test_selection = [' + ", ".join(f'"{t}"' for t in tests) + ']', s)
open("pyproject.toml", "w").write(s)
PY
mutmut run || true
mutmut results
echo "=== DIFFS"
for m in $(mutmut results 2>/dev/null | grep -E "survived|timeout" | sed -E "s/: .*//; s/^ +//"); do echo "##### $m"; mutmut show $m 2>&1 | grep -E "^[-+][^-+]"; done

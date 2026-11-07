# One command per test layer. Integration/chaos need a real Postgres+pgvector:
#   docker run -d --name citeguard-pgtest -e POSTGRES_USER=citeguard -e POSTGRES_PASSWORD=citeguard_password \
#     -e POSTGRES_DB=citeguard -p 5433:5432 pgvector/pgvector:pg16
#   export TEST_DATABASE_URL=postgresql+asyncpg://citeguard:citeguard_password@localhost:5433/citeguard
PY ?= python

eval:           ## recompute every README number from committed data -> evals/results.json + README table (no network)
	$(PY) scripts/make_results.py

test-unit:      ## no database, no network: unit + property + injection corpus
	$(PY) -m pytest tests -q -m "not live and not chaos and not int" --ignore=tests/pg

test-int:       ## real Postgres: schema, ledger, API
	$(PY) -m pytest tests/pg tests/api -q -m "not live"

test-sec:       ## SSRF + injection corpus (required CI check)
	$(PY) -m pytest tests/sec -q -m sec

test-chaos:     ## fault injection against a fake provider, real Postgres
	$(PY) -m pytest tests/chaos -q -m chaos

test-all:       ## everything except live
	$(PY) -m pytest -q -m "not live"

test-cov:
	$(PY) -m pytest -q -p no:randomly -m "not live" --cov=app.core.verifier --cov=app.core.llm --cov=app.core.fetcher --cov-branch --cov-report=term-missing

test-mutation:  ## mutmut in Linux Docker (mutmut does not run on native Windows)
	docker run --rm -v "$(CURDIR):/work" -w /work python:3.12-slim bash scripts/run_mutation.sh $(MUT_TARGET) "$(MUT_TESTS)"

import pytest


@pytest.fixture(autouse=True)
def _evidence_hmac_key(monkeypatch):
    # Test-only key; production reads CITEGUARD_EVIDENCE_HMAC_KEY from the environment.
    monkeypatch.setenv("CITEGUARD_EVIDENCE_HMAC_KEY", "test-key-not-a-secret")

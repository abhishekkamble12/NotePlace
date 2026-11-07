"""Aggregates evals/real/{dev,test}_q*.json (exported by run_eval_jobs.py) into a table. Pure reader; no API.
Shadow verdicts are MODEL OUTPUTS, not accuracy: no human label exists for them yet.
Usage: python scripts/summarize_real_runs.py"""
import glob
import json
from collections import Counter
from pathlib import Path

REAL = Path(__file__).resolve().parents[1] / "evals" / "real"


def summarize() -> dict:
    tot, lat, verd = Counter(), [], Counter()
    for f in sorted(REAL.glob("*_q*.json")):
        if f.name.endswith(".failed.json"):
            tot["failed_jobs"] += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        L = d["ledger"]
        tot["jobs"] += 1
        for k in ("llm_calls", "cache_hits", "tokens_in", "tokens_out", "unknown_usage_attempts", "failed_attempts"):
            tot[k] += L[k]
        tot["usd_modeled"] += L["usd_modeled"]
        lat += L["latency_ms"]
        for c in d["claims"]:
            tot["claims"] += 1
            tot["quote_not_found"] += not c["quote_found"]
            verd[c["shadow_verdict"] or "none"] += 1
    lat.sort()
    return {**tot, "usd_modeled": round(tot["usd_modeled"], 4), "shadow_verdicts": dict(verd),
            "call_latency_ms_p50": lat[len(lat) // 2] if lat else None,
            "call_latency_ms_p95": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else None, "n_latency": len(lat),
            "files": len(glob.glob(str(REAL / '*_q*.json')))}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=1))

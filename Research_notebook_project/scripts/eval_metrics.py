"""Shared eval statistics. Pure functions, no I/O, no LLM. Positive class = "kept" (verifier says `supports`).

precision = P(gold supports | kept)      <- what a reader of the report cares about
retention = P(kept | gold supports)      <- a.k.a. coverage/recall; always report next to precision
far       = P(kept | gold not supports)  <- false-acceptance rate, the primary safety metric; frr = 1 - retention
"""
import math
import random

LABELS = ["supports", "partial", "unsupported"]


def counts(gold: list[str], pred: list[str]) -> dict:
    tp = sum(g == "supports" and p == "supports" for g, p in zip(gold, pred, strict=True))
    kept, pos = sum(p == "supports" for p in pred), sum(g == "supports" for g in gold)
    precision, retention = (tp / kept if kept else math.nan), (tp / pos if pos else math.nan)
    f1 = 2 * precision * retention / (precision + retention) if tp else (0.0 if kept and pos else math.nan)
    neg, fa = len(gold) - pos, kept - tp
    return {"n": len(gold), "kept": kept, "gold_supports": pos, "tp": tp, "precision": precision, "retention": retention, "f1": f1,
            "gold_not_supports": neg, "false_accepts": fa, "far": fa / neg if neg else math.nan, "frr": 1 - retention}


def confusion(gold: list[str], pred: list[str]) -> dict:
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    for g, p in zip(gold, pred, strict=True):
        conf[g][p] += 1
    return conf


def bootstrap_ci(gold: list[str], pred: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile 95% CI of precision/retention/f1, resampling claims. Undefined resamples (no kept / no positives) are skipped
    and counted. shortcut: i.i.d. over claims; claims from one source/report are correlated so this is optimistic, use grouped resampling once groups are labeled."""
    rng, idx = random.Random(seed), range(len(gold))
    draws: dict[str, list[float]] = {"precision": [], "retention": [], "f1": [], "far": []}
    skipped = 0
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        c = counts([gold[i] for i in s], [pred[i] for i in s])
        if not math.isnan(c["far"]):  # FAR has its own denominator, so it is drawn even when precision is undefined
            draws["far"].append(c["far"])
        if math.isnan(c["precision"]) or math.isnan(c["retention"]):
            skipped += 1
            continue
        for k in ("precision", "retention", "f1"):
            draws[k].append(c[k])
    out: dict = {"n_boot": n_boot, "skipped_resamples": skipped}
    for k, v in draws.items():
        v.sort()
        out[k] = [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]] if v else [math.nan, math.nan]
    return out


def precision_coverage_sweep(gold: list[str], scores: list[float]) -> list[dict]:
    """Keep claims with score >= t for every distinct t. Rows: threshold, kept, precision, retention. No threshold is chosen here."""
    pos = sum(g == "supports" for g in gold)
    rows = []
    for t in sorted(set(scores)):
        kept = [g for g, s in zip(gold, scores, strict=True) if s >= t]
        tp = sum(g == "supports" for g in kept)
        rows.append({"threshold": t, "kept": len(kept), "precision": tp / len(kept) if kept else math.nan, "retention": tp / pos if pos else math.nan})
    return rows


def recall_at_k(ranked: list[list[str]], gold: list[set[str]], k: int) -> float:
    """Mean over queries of |gold found in top-k| / |gold|. Queries with empty gold are excluded."""
    vals = [len(set(r[:k]) & g) / len(g) for r, g in zip(ranked, gold, strict=True) if g]
    return sum(vals) / len(vals) if vals else math.nan


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson 95% interval for k/n. Unlike the percentile bootstrap it is not [1, 1] when every kept claim is correct."""
    if n == 0:
        return [math.nan, math.nan]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]

# scripts/nli_baseline.py
"""DeBERTa-MNLI baseline. Independent of the CiteGuard verifier: premise = snapshot span, hypothesis = claim,
score = P(entailment). Writes per-claim scores so eval can sweep thresholds. Needs: pip install -e '.[eval]'
(first run downloads microsoft/deberta-large-mnli, ~1.6 GB). Usage: python scripts/nli_baseline.py labeled.csv out.json"""
import csv
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "microsoft/deberta-large-mnli"


def main(src: str, dst: str) -> None:
    tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    ent = next(i for i, n in model.config.id2label.items() if n.lower() == "entailment")
    out = {}
    with open(src, encoding="utf-8") as f, torch.no_grad():
        for r in csv.DictReader(f):
            enc = tok(r["span"], r["claim"], return_tensors="pt", truncation=True, max_length=512)
            out[r["claim_id"]] = torch.softmax(model(**enc).logits, -1)[0, ent].item()
    json.dump({"model": MODEL, "entailment_prob": out}, open(dst, "w"), indent=1)
    print(f"{len(out)} scores -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

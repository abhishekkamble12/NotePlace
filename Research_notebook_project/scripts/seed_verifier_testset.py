# scripts/seed_verifier_testset.py
"""
Seeds the verifier_testset table from evals/verifier_testset.csv.
Idempotent: clears existing testset entries before inserting.
"""
import asyncio
import csv
import sys
from pathlib import Path

from sqlalchemy import delete

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import AsyncSessionLocal, init_db
from app.models import VerifierTestSet

CSV_PATH = Path(__file__).resolve().parents[1] / "evals" / "verifier_testset.csv"


async def seed_testset():
    await init_db()

    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run scripts/generate_testset.py first.")
        return

    async with AsyncSessionLocal() as session:
        # Clear existing entries
        await session.execute(delete(VerifierTestSet))

        count = 0
        with open(CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = VerifierTestSet(
                    quote=row["quote"],
                    claim=row["claim"],
                    context=row.get("context", ""),
                    perturbation_type=row.get("perturbation_type", "original"),
                    label=row["label"],
                    source=row.get("source", "")
                )
                session.add(entry)
                count += 1

        await session.commit()
        print(f"Successfully seeded {count} benchmark pairs into verifier_testset table.")


if __name__ == "__main__":
    asyncio.run(seed_testset())

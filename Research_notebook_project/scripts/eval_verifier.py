# scripts/eval_verifier.py
import asyncio
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import AsyncSessionLocal, init_db
from app.models import Job, VerifierTestSet
from app.verifier import verify_claim

async def run_evaluation():
    await init_db()

    async with AsyncSessionLocal() as session:
        # Create a mock job for the evaluation run
        eval_job = Job(
            question="Evaluation Benchmark Run for Verifier",
            max_tokens=200_000,
            max_minutes=30,
            mode="single",
            status="running",
            stage="verifying"
        )
        session.add(eval_job)
        await session.commit()
        await session.refresh(eval_job)

        # Fetch test set
        result = await session.execute(select(VerifierTestSet))
        items = result.scalars().all()

        if not items:
            print("No items found in verifier_testset. Run 'python scripts/seed_verifier_testset.py' first.")
            return

        print(f"\n=======================================================")
        print(f"  RUNNING CITEGUARD VERIFIER BENCHMARK ({len(items)} samples)")
        print(f"=======================================================\n")

        total = len(items)
        tp = 0  # True Positive: actual unsupported, predicted unsupported
        fp = 0  # False Positive: actual supported, predicted unsupported
        tn = 0  # True Negative: actual supported, predicted supported
        fn = 0  # False Negative: actual unsupported, predicted supported/partial

        type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for i, item in enumerate(items, 1):
            # snapshot is context + quote
            snapshot = f"{item.context}\n{item.quote}"
            res = await verify_claim(
                db=session,
                job_id=eval_job.id,
                claim_text=item.claim,
                quote=item.quote,
                snapshot=snapshot
            )

            is_actual_unsupported = item.label == "unsupported"
            is_pred_unsupported = res.verdict == "unsupported"

            if is_actual_unsupported and is_pred_unsupported:
                tp += 1
            elif not is_actual_unsupported and is_pred_unsupported:
                fp += 1
            elif not is_actual_unsupported and not is_pred_unsupported:
                tn += 1
            else:
                fn += 1

            type_stat = type_stats[item.perturbation_type]
            type_stat["total"] += 1
            # Check if prediction matched the exact label or direction
            if res.verdict == item.label or (item.label == "unsupported" and res.verdict == "unsupported"):
                type_stat["correct"] += 1

            print(f"[{i}/{total}] [{item.perturbation_type:<18}] Label: {item.label:<11} Pred: {res.verdict:<11} Match: {res.match_score:.0f}%")
            if res.reason:
                print(f"       Reason: {res.reason}")

        # Metrics calculation
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_drop_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print("\n" + "=" * 55)
        print("                 BENCHMARK RESULTS")
        print("=" * 55)
        print(f"Total Evaluated Claims       : {total}")
        print(f"Verifier Catch Rate (Recall) : {recall * 100:.1f}%")
        print(f"Citation Precision           : {precision * 100:.1f}%")
        print(f"False-Drop Rate              : {false_drop_rate * 100:.1f}%")
        print(f"F1 Score                     : {f1:.3f}")
        print("-" * 55)
        print(f"{'Perturbation Type':<25} | {'Accuracy':<10}")
        print("-" * 55)
        for p_type, stats in type_stats.items():
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
            print(f"{p_type:<25} | {acc:.1f}% ({stats['correct']}/{stats['total']})")
        print("=" * 55 + "\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation())


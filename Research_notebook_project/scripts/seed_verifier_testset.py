# scripts/seed_verifier_testset.py
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import AsyncSessionLocal, init_db
from app.models import VerifierTestSet

SAMPLE_BENCHMARK_PAIRS = [
    # 1. Quantum Computing
    {
        "quote": "In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "context": "Quantum computational supremacy marks a milestone where a programmable quantum device solves a problem no classical computer can feasibly compute. In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years. IBM later contested this estimate, suggesting classical algorithms could solve it in 2.5 days with ideal memory storage.",
        "claim": "Google demonstrated quantum supremacy in 2019 using their 53-qubit Sycamore processor.",
        "perturbation_type": "original",
        "label": "supports"
    },
    {
        "quote": "In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "context": "Quantum computational supremacy marks a milestone where a programmable quantum device solves a problem no classical computer can feasibly compute. In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "claim": "Google researchers used a 72-qubit Sycamore processor to achieve quantum supremacy in 2019.",
        "perturbation_type": "number_changed",
        "label": "unsupported"
    },
    {
        "quote": "In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "context": "Quantum computational supremacy marks a milestone. In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "claim": "In 2023, Google's Sycamore processor solved a problem that would take classical supercomputers 10,000 years.",
        "perturbation_type": "date_changed",
        "label": "unsupported"
    },
    {
        "quote": "In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "context": "In 2019, Google researchers announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "claim": "Google's Sycamore processor permanently obsoleted all classical supercomputers across all computing tasks.",
        "perturbation_type": "overgeneralized",
        "label": "unsupported"
    },

    # 2. Electric Vehicles & Batteries
    {
        "quote": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging.",
        "context": "Battery innovation continues to accelerate. CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging. Commercial rollouts in production vehicles are scheduled to begin late this year.",
        "claim": "CATL's Shenxing Plus LFP battery claims to deliver over 1,000 km of range.",
        "perturbation_type": "original",
        "label": "supports"
    },
    {
        "quote": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging.",
        "context": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging.",
        "claim": "CATL's Shenxing Plus battery was proven independently on public roads to achieve 1,200 km range.",
        "perturbation_type": "unsupported_claim",
        "label": "unsupported"
    },
    {
        "quote": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging.",
        "context": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range of over 1,000 kilometers and the ability to add 600 km of range in 10 minutes of fast charging.",
        "claim": "CATL announced a fast-charging LFP battery in 2024 that can recharge rapidly under specific lab conditions.",
        "perturbation_type": "slight_qualifier",
        "label": "partial"
    },

    # 3. AI & Scaling Laws
    {
        "quote": "Kaplan et al. (2020) observed that cross-entropy loss scales as a power law with compute, dataset size, and parameter count, with performance depending strongly on scale and weakly on model shape.",
        "context": "Empirical analysis of neural language models showed predictable scaling laws. Kaplan et al. (2020) observed that cross-entropy loss scales as a power law with compute, dataset size, and parameter count, with performance depending strongly on scale and weakly on model shape.",
        "claim": "Kaplan et al. found that loss scales as a power law with model parameters, compute, and dataset size.",
        "perturbation_type": "original",
        "label": "supports"
    },
    {
        "quote": "Kaplan et al. (2020) observed that cross-entropy loss scales as a power law with compute, dataset size, and parameter count, with performance depending strongly on scale and weakly on model shape.",
        "context": "Kaplan et al. (2020) observed that cross-entropy loss scales as a power law with compute, dataset size, and parameter count, with performance depending strongly on scale and weakly on model shape.",
        "claim": "Kaplan et al. proved that increasing dataset size alone causes models to become conscious and eliminate hallucination.",
        "perturbation_type": "causal_leap",
        "label": "unsupported"
    },

    # 4. Hallucinated / Non-existent Quote in Source
    {
        "quote": "Global semiconductor revenues reached 526 billion dollars in 2023, representing an 8 percent decrease compared to 2022.",
        "context": "Semiconductor industry revenues fluctuate with cyclical consumer electronics demand. Many chip manufacturers faced elevated inventory levels throughout the first half of the year.",
        "claim": "Semiconductor sales declined by 8% in 2023 to $526 billion.",
        "perturbation_type": "fabricated_quote",
        "label": "unsupported" # Because quote does not exist in context/snapshot!
    }
]

async def seed():
    await init_db()
    async with AsyncSessionLocal() as session:
        for item in SAMPLE_BENCHMARK_PAIRS:
            entry = VerifierTestSet(
                quote=item["quote"],
                context=item["context"],
                claim=item["claim"],
                perturbation_type=item["perturbation_type"],
                label=item["label"]
            )
            session.add(entry)
        await session.commit()
        print(f"Successfully seeded {len(SAMPLE_BENCHMARK_PAIRS)} benchmark verification pairs into verifier_testset.")

if __name__ == "__main__":
    asyncio.run(seed())


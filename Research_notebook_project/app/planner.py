# app/planner.py
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import call_llm
from app.models import SubQuestion


class ResearchPlan(BaseModel):
    subquestions: list[str] = Field(
        min_length=4,
        max_length=8,
        description="List of 4 to 8 targeted, distinct sub-questions that collectively answer the main question."
    )

PLANNER_SYSTEM_PROMPT = """You are an expert research strategist.
Your task is to decompose a broad research question into 4 to 8 specific, highly targeted sub-questions.

Guidelines:
1. Cover factual background, comparative analysis, mechanisms, and key controversies/challenges.
2. Ensure each sub-question can be researched independently using web sources.
3. Avoid overlapping or redundant questions.
4. Keep each sub-question clear, neutral, and precise."""

async def generate_plan(
    db: AsyncSession,
    job_id: int,
    main_question: str,
    planner_model: str = "openai/gpt-oss-120b"
) -> list[SubQuestion]:
    """
    Decomposes the main research question into 4-8 subquestions using call_llm
    under the 'planning' phase and saves them to the database.
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Decompose this research question into 4 to 8 sub-questions:\n\"{main_question}\""}
    ]

    plan = await call_llm(
        scope=("job", job_id),
        phase="planning",
        model=planner_model,
        messages=messages,
        max_tokens=600,
        schema=ResearchPlan,
        db=db
    )

    if not plan or not plan.subquestions:
        # Fallback if planning LLM fails
        default_subqs = [
            f"What is the background and current status of {main_question}?",
            f"What are the primary technical or economic arguments regarding {main_question}?",
            f"What are the major counterarguments or disagreements concerning {main_question}?",
            f"What are the future outlooks and consensus conclusions on {main_question}?"
        ]
        subq_texts = default_subqs
    else:
        subq_texts = plan.subquestions

    created_subquestions: list[SubQuestion] = []
    for text in subq_texts:
        subq = SubQuestion(
            job_id=job_id,
            text=text,
            status="pending"
        )
        db.add(subq)
        created_subquestions.append(subq)

    await db.commit()
    for subq in created_subquestions:
        await db.refresh(subq)

    return created_subquestions

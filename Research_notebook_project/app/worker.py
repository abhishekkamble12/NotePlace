# app/worker.py
import os
from arq.connections import RedisSettings
from app.db import init_db
from app.pipeline import execute_research_job

# Redis settings matching docker-compose
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

async def startup(ctx: dict) -> None:
    """Initializes database schema and connections when worker boots."""
    await init_db()

async def shutdown(ctx: dict) -> None:
    """Clean shutdown handler."""
    pass

async def run_job_task(ctx: dict, job_id: int) -> None:
    """Worker task wrapper for research jobs."""
    await execute_research_job(job_id)

class WorkerSettings:
    """Arq worker configuration."""
    functions = [run_job_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
    max_jobs = 10
    job_timeout = 600  # 10 minutes max per job


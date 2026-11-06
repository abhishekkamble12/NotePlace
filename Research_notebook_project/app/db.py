# app/db.py
import asyncio
import os
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base

# Database connection URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://citeguard:citeguard_password@localhost:5432/citeguard"
)

# Async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# FastAPI session dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def run_alembic_migrations() -> None:
    """Run alembic upgrade head."""
    from alembic.config import Config

    from alembic import command
    base_dir = os.path.dirname(os.path.dirname(__file__))
    alembic_ini_path = os.path.join(base_dir, "alembic.ini")
    if os.path.exists(alembic_ini_path):
        alembic_cfg = Config(alembic_ini_path)
        alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
        command.upgrade(alembic_cfg, "head")

# Helper to initialize pgvector & run migrations on startup
async def init_db() -> None:
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception:
        pass

    try:
        await asyncio.to_thread(run_alembic_migrations)
    except Exception:
        # Fallback to create_all for test/isolated database setups
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

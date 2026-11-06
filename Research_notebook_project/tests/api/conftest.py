import asyncio
import os
import uuid

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command

BASE_URL = os.getenv("TEST_DATABASE_URL")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


async def _admin(sql):
    eng = create_async_engine(BASE_URL.rsplit("/", 1)[0] + "/postgres", isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with eng.connect() as c:
        await c.execute(text(sql))
    await eng.dispose()


@pytest.fixture
def pgdb(monkeypatch):
    """Throwaway REAL Postgres+pgvector database migrated to head (sync fixture: alembic runs its own event loop)."""
    if not BASE_URL:
        pytest.skip("TEST_DATABASE_URL not set: real-Postgres tests NOT run")
    name = "ct_" + uuid.uuid4().hex[:10]
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    url = BASE_URL.rsplit("/", 1)[0] + "/" + name
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "alembic"))
    command.upgrade(cfg, "head")
    yield url
    asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@pytest.fixture
def sm_factory(pgdb):
    def make():
        eng = create_async_engine(pgdb, poolclass=NullPool)
        return async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    return make

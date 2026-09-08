import os
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def database(monkeypatch):
    admin_url = os.getenv("APP_TEST_DATABASE_URL")
    if not admin_url:
        pytest.skip("Run bin/integration-test with the disposable Postgres container")
    admin = await asyncpg.connect(admin_url.replace("postgresql+asyncpg", "postgresql"))
    name = "platform_test_" + uuid4().hex
    await admin.execute(f'CREATE DATABASE "{name}"')
    url = admin_url.rsplit("/", 1)[0] + "/" + name
    monkeypatch.setenv("APP_DATABASE_URL", url)
    monkeypatch.setenv("APP_ENV", "test")
    try:
        yield url
    finally:
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()

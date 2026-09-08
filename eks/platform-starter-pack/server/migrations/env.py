import asyncio
import ssl

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from app import models  # noqa: F401
from app.settings import Settings


def migrate(connection):
    context.configure(connection=connection, target_metadata=SQLModel.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def online():
    settings = Settings()
    engine = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,
        connect_args={"ssl": ssl.create_default_context(cafile=settings.postgres_ca_file)}
        if settings.postgres_ca_file
        else {},
    )
    try:
        async with engine.connect() as connection:
            async with connection.begin():
                await connection.exec_driver_sql("SET LOCAL lock_timeout = '30s'")
                await connection.exec_driver_sql("SELECT pg_advisory_xact_lock(72641001)")
                await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url, target_metadata=SQLModel.metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())

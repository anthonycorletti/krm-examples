import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.settings import Settings


def create_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={
            "timeout": 3,
            "command_timeout": 5,
            "server_settings": {"statement_timeout": "5000"},
            **(
                {"ssl": ssl.create_default_context(cafile=settings.postgres_ca_file)}
                if settings.postgres_ca_file
                else {}
            ),
        },
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def transaction(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with create_sessionmaker(engine)() as session:
        async with session.begin():
            yield session

from functools import cache

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select

from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.kit.db.postgres import create_sessionmaker, transaction
from app.models import Verification
from app.verification.schemas import MigrationRead, VerificationRead


@cache
def expected_revision() -> str:
    head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    if head is None:
        raise RuntimeError("Migration head is missing")
    return head


class VerificationService:
    def __init__(self, engine: AsyncEngine, auth: AuthService):
        self.engine, self.auth = engine, auth
        self.expected = expected_revision()

    async def run(self, principal: Principal, environment: str) -> VerificationRead:
        self.auth.authorize(principal, "platform:verify", environment)
        async with transaction(self.engine) as session:
            record = Verification(
                environment=environment,
                subject=principal.subject,
                evidence="Committed a verification record and read it from a new database session",
            )
            session.add(record)
            await session.flush()
            identifier = record.id
        async with create_sessionmaker(self.engine)() as session:
            persisted = await session.get(Verification, identifier)
            if persisted is None:
                raise RuntimeError("Committed record was not readable")
            return VerificationRead.model_validate(persisted, from_attributes=True)

    async def history(self, principal: Principal, environment: str) -> list[VerificationRead]:
        self.auth.authorize(principal, "platform:read", environment)
        async with create_sessionmaker(self.engine)() as session:
            result = await session.exec(
                select(Verification)
                .where(Verification.environment == environment)
                .order_by(col(Verification.created_at).desc())
                .limit(50)
            )
            return [VerificationRead.model_validate(r, from_attributes=True) for r in result.all()]

    async def migrations(self, principal: Principal, environment: str) -> MigrationRead:
        self.auth.authorize(principal, "platform:read", environment)
        async with self.engine.connect() as connection:
            applied = await connection.run_sync(
                lambda conn: MigrationContext.configure(conn).get_current_revision()
            )
        return MigrationRead(
            applied=applied, expected=self.expected, current=applied == self.expected
        )

from datetime import datetime

from sqlmodel import SQLModel


class VerificationCreate(SQLModel):
    environment: str = "local"


class VerificationRead(SQLModel):
    id: str
    created_at: datetime
    environment: str
    component: str
    subject: str
    evidence: str


class MigrationRead(SQLModel):
    applied: str | None
    expected: str
    current: bool

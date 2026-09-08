from datetime import datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel
from ulid import ULID

from app.kit.utils import utc_now


class Timestamp(DateTime):
    def __init__(self, timezone: bool = True):
        super().__init__(timezone=timezone)


class RecordModel(SQLModel):
    id: str = Field(default_factory=lambda: str(ULID()), primary_key=True)
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=Timestamp,
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_type=Timestamp,
        nullable=False,
        sa_column_kwargs={"onupdate": utc_now},
    )
    deleted_at: datetime | None = Field(default=None, sa_type=Timestamp, nullable=True, index=True)

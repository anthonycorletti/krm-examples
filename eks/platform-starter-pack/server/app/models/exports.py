from sqlalchemy import JSON
from sqlmodel import Field
from ulid import ULID

from app.kit.db.models.base import RecordModel


class Export(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"exp_{ULID()}", primary_key=True)
    owner: str = Field(index=True)
    environment: str = Field(index=True)
    request_key: str = Field(unique=True)
    trigger: str = "manual"
    status: str = Field(default="queued", index=True)
    workflow_name: str | None = None
    manifest_key: str | None = None
    row_counts: dict[str, int] = Field(default_factory=dict, sa_type=JSON)
    error: str | None = None

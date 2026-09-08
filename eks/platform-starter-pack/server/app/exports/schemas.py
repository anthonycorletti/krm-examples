from sqlmodel import Field, SQLModel

from app.models import Export


class ExportCreate(SQLModel):
    request_key: str = Field(min_length=1, max_length=128)


class WarehouseRead(SQLModel):
    export: Export | None
    counts: dict[str, int]

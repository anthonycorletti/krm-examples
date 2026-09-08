from typing import Literal

from sqlmodel import SQLModel


class ComponentRead(SQLModel):
    id: str
    name: str
    status: Literal["ready", "unavailable", "unverified", "not_configured", "not_implemented"]
    detail: str

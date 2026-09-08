from sqlmodel import SQLModel


class Principal(SQLModel):
    subject: str
    environment: str
    scopes: list[str]


class AuthConfig(SQLModel):
    environment: str
    issuer: str | None
    client_id: str | None

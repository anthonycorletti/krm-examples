from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")
    env: Literal["local", "test", "preview", "production"] = "local"
    database_url: str = "postgresql+asyncpg://platform:platform@127.0.0.1:5432/platform"
    token_secret: SecretStr = SecretStr("")
    issuer: str = "platform-local"
    api_audience: str = "platform-api"
    mcp_audience: str = "platform-mcp"
    oidc_jwks_url: str = ""
    oidc_client_id: str = ""
    postgres_ca_file: str = ""
    object_endpoint: str = ""
    object_bucket: str = "platform"
    object_access_key: str = ""
    object_secret_key: SecretStr = SecretStr("")
    valkey_url: SecretStr = SecretStr("")
    agent_model: str = ""
    otel_endpoint: str = ""
    worker_image: str = "platform-starter-server:local"
    workflow_namespace: str = "platform-apps"

    @field_validator("database_url")
    @classmethod
    def async_url(cls, value: str) -> str:
        url = make_url(value)
        if url.drivername not in {"postgres", "postgresql", "postgresql+asyncpg"}:
            raise ValueError("PostgreSQL with asyncpg is required")
        return url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)

    @model_validator(mode="after")
    def validate_auth(self):
        if self.env not in {"local", "test"} and not self.oidc_jwks_url:
            raise ValueError("OIDC is required outside local/test")
        if self.oidc_jwks_url and (
            not self.oidc_jwks_url.startswith("https://") or not self.issuer.startswith("https://")
        ):
            raise ValueError("OIDC issuer and JWKS endpoints must use HTTPS")
        return self

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.schemas import AuthConfig, Principal
from app.kit.errors import Unauthorized

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


@router.get("/config", operation_id="read_auth_config")
async def config(request: Request) -> AuthConfig:
    settings = request.app.state.settings
    return AuthConfig(
        environment=settings.env,
        issuer=settings.issuer if settings.oidc_jwks_url else None,
        client_id=settings.oidc_client_id or None,
    )


async def current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    if credentials is None:
        raise Unauthorized("Bearer token required")
    return await request.app.state.auth.authenticate(
        credentials.credentials,
        request.app.state.settings.api_audience,
    )


@router.get("/me", operation_id="read_identity")
async def me(principal: Principal = Depends(current_principal)) -> Principal:
    return principal

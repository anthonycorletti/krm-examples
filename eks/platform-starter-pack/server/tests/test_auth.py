import time

import jwt
import pytest

from app.auth.service import AuthService
from app.kit.errors import Forbidden, Unauthorized
from app.mcp import SharedTokenVerifier
from app.settings import Settings

SECRET = "test-only-signing-key-with-at-least-32-characters"


def token(
    audience="platform-api", scope="platform:read", environment="test", expiry=60, sub="test-user"
):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "iss": "platform-local",
            "aud": audience,
            "iat": now,
            "exp": now + expiry,
            "scope": scope,
            "environment": environment,
        },
        SECRET,
        algorithm="HS256",
    )


async def test_auth_denies_environment_and_scope():
    auth = AuthService(Settings(env="test", token_secret=SECRET))
    principal = await auth.authenticate(token(), "platform-api")
    auth.authorize(principal, "platform:read", "test")
    with pytest.raises(Forbidden):
        auth.authorize(principal, "platform:verify", "test")
    with pytest.raises(Forbidden):
        auth.authorize(principal, "platform:read", "production")


@pytest.mark.parametrize("value", [token(expiry=-10), token(audience="platform-mcp"), "invalid"])
async def test_invalid_tokens_are_rejected(value):
    auth = AuthService(Settings(env="test", token_secret=SECRET))
    with pytest.raises(Unauthorized):
        await auth.authenticate(value, "platform-api")


async def test_mcp_verifies_its_own_audience():
    verifier = SharedTokenVerifier(AuthService(Settings(env="test", token_secret=SECRET)))
    assert await verifier.verify_token(token()) is None
    assert await verifier.verify_token(token(audience="platform-mcp")) is not None


@pytest.mark.parametrize("environment", ["preview", "production"])
def test_development_auth_cannot_start_remotely(environment):
    with pytest.raises(ValueError, match="OIDC is required"):
        Settings(env=environment)

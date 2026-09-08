import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.auth.service import AuthService
from app.kit.errors import Forbidden, Unauthorized
from app.mcp import SharedTokenVerifier
from app.settings import Settings


async def test_oidc_shared_validation_and_key_cache(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update(kid="key-1", use="sig", alg="RS256")
    requests = []

    def keys(request):
        requests.append(request.url)
        return httpx.Response(200, json={"keys": [jwk]})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(keys), **kwargs),
    )
    settings = Settings(
        env="production",
        issuer="https://identity.example",
        oidc_jwks_url="https://identity.example/keys",
    )
    auth = AuthService(settings)

    def token(audience="platform-api", **overrides):
        return jwt.encode(
            dict(
                sub="user",
                iss=settings.issuer,
                aud=audience,
                iat=int(time.time()),
                exp=int(time.time()) + 60,
                environment="production",
                scope="platform:read",
            )
            | overrides,
            private,
            algorithm="RS256",
            headers={"kid": "key-1"},
        )

    principal = await auth.authenticate(token(), settings.api_audience)
    auth.authorize(principal, "platform:read", "production")
    with pytest.raises(Forbidden):
        auth.authorize(principal, "platform:verify", "production")
    with pytest.raises(Forbidden):
        auth.authorize(principal, "platform:read", "preview")
    verifier = SharedTokenVerifier(auth)
    assert await verifier.verify_token(token()) is None
    assert await verifier.verify_token(token("platform-mcp")) is not None
    for invalid in [
        token(iss="https://other.example"),
        token(exp=1),
        token(scope=[]),
        jwt.encode({}, "local-secret", algorithm="HS256"),
    ]:
        with pytest.raises(Unauthorized):
            await auth.authenticate(invalid, settings.api_audience)
    assert len(requests) == 1


async def test_oidc_unavailable_fails_closed(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(transport=transport, **kw))
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode({}, private, algorithm="RS256", headers={"kid": "missing"})
    auth = AuthService(
        Settings(
            env="production",
            issuer="https://identity.example",
            oidc_jwks_url="https://identity.example/keys",
        )
    )
    with pytest.raises(Unauthorized):
        await auth.authenticate(token, "platform-api")


def test_postgres_uri_preserves_encoded_credentials():
    from sqlalchemy.engine import make_url

    settings = Settings(database_url="postgresql://platform:p%40ss%2Fword@db/platform")
    url = make_url(settings.database_url)
    assert url.drivername == "postgresql+asyncpg"
    assert url.password == "p@ss/word"

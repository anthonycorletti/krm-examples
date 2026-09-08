import asyncio
import time

import httpx
import jwt

from app.auth.schemas import Principal
from app.kit.errors import Forbidden, Unauthorized
from app.settings import Settings


class AuthService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._keys: dict[str, jwt.PyJWK] = {}
        self._refreshed = float("-inf")
        self._lock = asyncio.Lock()

    async def _signing_key(self, token: str):
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if header.get("alg") != "RS256" or not isinstance(kid, str):
            raise Unauthorized("Unsupported signing key")
        async with self._lock:
            age = time.monotonic() - self._refreshed
            if age > 300 or (kid not in self._keys and age > 30):
                # Unknown key IDs cannot trigger more than one fetch every 30 seconds.
                self._refreshed = time.monotonic()
                async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
                    response = await client.get(self.settings.oidc_jwks_url)
                    response.raise_for_status()
                    payload = response.json()
                self._keys = {
                    key["kid"]: jwt.PyJWK.from_dict(key, algorithm="RS256")
                    for key in payload["keys"]
                    if key.get("kty") == "RSA"
                    and key.get("use", "sig") == "sig"
                    and key.get("alg", "RS256") == "RS256"
                    and isinstance(key.get("kid"), str)
                }
            if kid not in self._keys:
                raise Unauthorized("Unknown signing key")
            return self._keys[kid].key

    async def authenticate(self, token: str, audience: str) -> Principal:
        try:
            if self.settings.oidc_jwks_url:
                key = await self._signing_key(token)
                algorithms = ["RS256"]
            else:
                key = self.settings.token_secret.get_secret_value()
                algorithms = ["HS256"]
                if len(key) < 32:
                    raise Unauthorized("Authentication is not configured")
            claims = jwt.decode(
                token,
                key,
                algorithms=algorithms,
                audience=audience,
                issuer=self.settings.issuer,
                options={"require": ["exp", "iat", "sub", "iss", "aud", "environment", "scope"]},
            )
            if not isinstance(claims["scope"], str):
                raise ValueError("Invalid scope")
            return Principal(
                subject=claims["sub"],
                environment=claims["environment"],
                scopes=claims["scope"].split(),
            )
        except (jwt.InvalidTokenError, ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
            raise Unauthorized("Invalid or expired token") from exc

    def authorize(self, principal: Principal, scope: str, environment: str):
        if environment != self.settings.env or principal.environment != environment:
            raise Forbidden("Environment access denied")
        if scope not in principal.scopes:
            raise Forbidden("Permission denied")

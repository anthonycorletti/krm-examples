from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.settings import Settings


class Cache:
    def __init__(self, settings: Settings):
        self.client = (
            Redis.from_url(
                settings.valkey_url.get_secret_value(),
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            if settings.valkey_url.get_secret_value()
            else None
        )

    async def progress(self, run_id: str, status: str):
        if self.client:
            try:
                await self.client.set(f"run:{run_id}:progress", status, ex=300)
                await self.client.publish(f"run:{run_id}", status)
            except (RedisError, OSError):
                pass  # SQL remains authoritative when the cache is unavailable.

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def read_progress(self, run_id: str) -> str | None:
        if self.client:
            try:
                value = await self.client.get(f"run:{run_id}:progress")
                return value.decode() if isinstance(value, bytes) else value
            except (RedisError, OSError):
                pass
        return None

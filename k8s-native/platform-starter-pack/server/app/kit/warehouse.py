import httpx


class Warehouse:
    def __init__(self, settings):
        self.settings = settings

    async def request(self, query, body=b""):
        if not self.settings.clickhouse_url:
            raise RuntimeError("Warehouse is not configured")
        async with httpx.AsyncClient(
            timeout=60,
            auth=(
                self.settings.clickhouse_user,
                self.settings.clickhouse_password.get_secret_value(),
            ),
        ) as client:
            response = await client.post(
                self.settings.clickhouse_url, params={"query": query}, content=body
            )
            response.raise_for_status()
            return response

    async def query(self, query):
        return (await self.request(query)).json()

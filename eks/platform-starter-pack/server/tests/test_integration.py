import asyncio
import socket
import sys

import httpx
import uvicorn
from fastmcp import Client
from pydantic_ai.models.test import TestModel

from app.agents.service import create_inspector
from app.api import create_app
from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.components.service import ComponentsService
from app.kit.db.postgres import create_db_engine
from app.mcp import create_mcp
from app.settings import Settings
from app.worker import inspect_demo
from tests.test_auth import SECRET, token


async def alembic(*args):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "alembic",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    assert process.returncode == 0, output.decode()


async def test_migration_and_api_evidence(database):
    settings = Settings(env="test", database_url=database, token_secret=SECRET)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503
            await alembic("upgrade", "head")
            assert (await client.get("/readyz")).status_code == 200
            assert (await client.get("/api/components")).status_code == 401
            client.headers["Authorization"] = "Bearer " + token(
                scope="platform:read platform:verify"
            )
            assert (await client.get("/api/components?environment=production")).status_code == 403
            response = await client.post("/api/verifications", json={"environment": "test"})
            assert response.status_code == 201, response.text
            identifier = response.json()["id"]
            await alembic("upgrade", "head")
            await alembic("check")
            history = await client.get("/api/verifications?environment=test")
            assert history.json()[0]["id"] == identifier
            client.headers["Authorization"] = "Bearer " + token()
            assert (
                await client.post("/api/verifications", json={"environment": "test"})
            ).status_code == 403
    # A new application instance reads the committed record.
    restarted = create_app(settings)
    async with restarted.router.lifespan_context(restarted):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted),
            base_url="http://test",
            headers={"Authorization": "Bearer " + token()},
        ) as client:
            assert (await client.get("/api/verifications?environment=test")).json()[0][
                "id"
            ] == identifier
    await alembic("downgrade", "base")
    await alembic("upgrade", "head")


async def test_mcp_http_auth_and_agent(database):
    await alembic("upgrade", "head")
    settings = Settings(env="test", database_url=database, token_secret=SECRET)
    mcp = create_mcp(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.02)
        assert server.started
        url = f"http://127.0.0.1:{port}/mcp"
        async with httpx.AsyncClient() as http:
            response = await http.post(url, headers={"Authorization": "Bearer " + token()}, json={})
            assert response.status_code == 401
        async with Client(url, auth=token(audience="platform-mcp")) as client:
            names = {tool.name for tool in await client.list_tools()}
            assert {
                "list_components",
                "create_project",
                "create_task",
                "read_task",
                "send_instruction",
            } <= names
            result = await client.call_tool("list_components", {"environment": "test"})
            assert not result.is_error
            denied = await client.call_tool(
                "run_verification", {"environment": "test"}, raise_on_error=False
            )
            assert denied.is_error
            denied = await client.call_tool(
                "list_components", {"environment": "production"}, raise_on_error=False
            )
            assert denied.is_error
        async with Client(
            url, auth=token(audience="platform-mcp", scope="platform:read platform:verify")
        ) as client:
            result = await client.call_tool("run_verification", {"environment": "test"})
            assert not result.is_error
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)
        sock.close()
    engine = create_db_engine(settings)
    try:
        agent = create_inspector(TestModel(), ComponentsService(engine, AuthService(settings)))
        result = await agent.run(
            "Inspect the platform",
            deps=Principal(
                subject="agent-test",
                environment="test",
                scopes=["platform:read"],
            ),
        )
        assert result.output
        assert any(message.kind == "request" for message in result.all_messages())
    finally:
        await engine.dispose()

    demo = await inspect_demo(settings)
    assert demo["environment"] == "test"
    assert "deterministic demo" in demo["model"]
    assert "SELECT 1 succeeded" in demo["output"]
    assert "Pydantic AI workflows" in demo["output"]

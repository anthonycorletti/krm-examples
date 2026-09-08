from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token

from app.auth.service import AuthService
from app.components.service import ComponentsService
from app.exports.service import ExportsService
from app.kit.db.postgres import create_db_engine
from app.kit.errors import Unauthorized
from app.kit.objects import Objects
from app.projects.schemas import ProjectCreate
from app.settings import Settings
from app.tasks.schemas import TaskCreate
from app.tasks.service import TasksService
from app.verification.service import VerificationService


class SharedTokenVerifier(TokenVerifier):
    def __init__(self, auth):
        super().__init__()
        self.auth = auth

    async def verify_token(self, token):
        try:
            principal = await self.auth.authenticate(token, self.auth.settings.mcp_audience)
        except Unauthorized:
            return None
        return AccessToken(token=token, client_id=principal.subject, scopes=principal.scopes)


def create_mcp(settings: Settings | None = None):
    settings = settings or Settings()
    auth = AuthService(settings)
    engine = create_db_engine(settings)
    components = ComponentsService(engine, auth)
    exports = ExportsService(engine, auth)
    verification = VerificationService(engine, auth)
    tasks = TasksService(engine, auth, Objects(settings))

    @asynccontextmanager
    async def lifespan(server):
        if not settings.oidc_jwks_url and len(settings.token_secret.get_secret_value()) < 32:
            raise RuntimeError("APP_TOKEN_SECRET must contain at least 32 characters")
        try:
            yield {}
        finally:
            await engine.dispose()

    server = FastMCP("Platform inspector", auth=SharedTokenVerifier(auth), lifespan=lifespan)

    async def principal():
        token = get_access_token()
        if token is None:
            raise Unauthorized("Bearer token required")
        return await auth.authenticate(token.token, settings.mcp_audience)

    @server.tool()
    async def list_components(environment: str = "local"):
        return await components.list(await principal(), environment)

    @server.tool()
    async def run_verification(environment: str = "local"):
        return await verification.run(await principal(), environment)

    @server.tool()
    async def list_verifications(environment: str = "local"):
        return await verification.history(await principal(), environment)

    @server.tool()
    async def read_migrations(environment: str = "local"):
        return await verification.migrations(await principal(), environment)

    @server.tool()
    async def list_exports():
        return await exports.list_exports(await principal())

    @server.tool()
    async def export_now(request_key: str):
        return await exports.create_export(await principal(), request_key)

    @server.tool()
    async def read_warehouse():
        return await exports.warehouse(await principal())

    @server.tool()
    async def list_projects():
        return await tasks.list(await principal())

    @server.tool()
    async def create_project(name: str, description: str = ""):
        return await tasks.create(
            await principal(), ProjectCreate(name=name, description=description)
        )

    @server.tool()
    async def list_tasks(project_id: str):
        return await tasks.list_tasks(await principal(), project_id)

    @server.tool()
    async def create_task(project_id: str, title: str, prompt: str, agent: str = "inspector"):
        return await tasks.create_task(
            await principal(), project_id, TaskCreate(title=title, prompt=prompt, agent=agent)
        )

    @server.tool()
    async def read_task(task_id: str):
        return await tasks.detail(await principal(), task_id)

    @server.tool()
    async def send_instruction(task_id: str, content: str):
        return await tasks.send(await principal(), task_id, content)

    @server.tool()
    async def cancel_task(task_id: str):
        await tasks.cancel(await principal(), task_id)
        return {"status": "cancelled"}

    return server


def create_http_app():
    return create_mcp().http_app()

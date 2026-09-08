import asyncio
from contextlib import asynccontextmanager
from time import monotonic

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.auth.service import AuthService
from app.components.service import ComponentsService
from app.kit.db.postgres import create_db_engine
from app.kit.errors import Forbidden, Unauthorized
from app.kit.metrics import LATENCY, REQUESTS
from app.kit.objects import Objects
from app.kit.telemetry import configure
from app.projects.service import ProjectsService
from app.router import router
from app.settings import Settings
from app.tasks.service import TasksService
from app.verification.service import VerificationService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        if not settings.oidc_jwks_url and len(settings.token_secret.get_secret_value()) < 32:
            raise RuntimeError(
                "Set APP_TOKEN_SECRET to a generated secret of at least 32 characters"
            )
        engine = create_db_engine(settings)
        telemetry = configure(settings, "platform-api")
        auth = AuthService(settings)
        app.state.settings, app.state.engine, app.state.auth = settings, engine, auth
        app.state.components = ComponentsService(engine, auth)
        app.state.verification = VerificationService(engine, auth)
        app.state.projects = ProjectsService(engine, auth)
        app.state.tasks = TasksService(engine, auth, Objects(settings))
        try:
            yield
        finally:
            await engine.dispose()
            if telemetry:
                await asyncio.to_thread(telemetry.shutdown)

    app = FastAPI(title="Experimentation platform API", version=__version__, lifespan=lifespan)
    app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = monotonic()
        with trace.get_tracer("platform.api").start_as_current_span("http.request") as span:
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", response.status_code)
            REQUESTS.labels(request.method, route, str(response.status_code)).inc()
            LATENCY.labels(route).observe(monotonic() - started)
            return response

    @app.exception_handler(Unauthorized)
    async def unauthorized(request: Request, exc: Unauthorized):
        return JSONResponse(
            {"detail": str(exc)}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )

    @app.exception_handler(Forbidden)
    async def forbidden(request: Request, exc: Forbidden):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    return app


app = create_app()

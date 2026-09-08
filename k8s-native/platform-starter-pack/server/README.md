# Server

Async FastAPI and FastMCP entrypoints share topic services, authorization, SQLModel metadata, and asyncpg sessions. Pydantic AI agents live in app/agents/; app/worker.py supplies dispatcher and run entrypoints for Argo. Python configuration, lockfile, tests, and Alembic migrations live here.

Projects and tasks enforce owner/environment access. Postgres stores durable run state and object references; SeaweedFS stores conversation bodies/history/reports; Valkey holds disposable progress. The dispatcher serializes queued runs within each task and reconciles deterministic Argo workflow names. Workers check cancellation and reject late completion. API and MCP reuse the same services.

The deterministic inspector calls real platform tools without model credentials. Set APP_AGENT_MODEL and supply the selected provider's credentials to agent workers to enable model-backed agents; this path still requires live validation. OpenTelemetry excludes model message contents. The API provides Prometheus request metrics.

Run commands from the example root:

- `bin/server-install`: frozen uv lockfile installation.
- `bin/server-check`: Ruff, ty, and unit tests.
- `bin/integration-test`: disposable Postgres databases, migrations, HTTP MCP, ownership, run concurrency, cancellation, and conversation tests.
- `bin/local-experiment`: live local task/follow-up exercise through Kubernetes services.
- `bin/migration-new "description"`, `bin/migrate`, `bin/migration-check`: reviewed Alembic changes and drift checks.
- `bin/server-build`: server container image from the pinned Astral GHCR Trixie image.

Native development uses `bin/dev-init`, `bin/postgres-start`, `bin/migrate`, `bin/server-dev`, and `bin/mcp-dev`; that database is separate from local Kubernetes. Native tokens come from bin/dev-token, while local K3s cluster tokens come from bin/local-token. Local symmetric tokens are only for local/test; remote environments require OIDC configuration. TLS terminates at Envoy in Kubernetes and is supplied by local bin/ tooling for native processes, never generated inside app/.

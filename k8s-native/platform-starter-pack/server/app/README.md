# Application package

api.py composes FastAPI; mcp.py adapts the same services into tools. router.py aggregates topic routers. settings.py defines environment configuration. Topic routers are thin async transport adapters; services enforce policy and await database operations. Schemas are non-table SQLModel contracts, and concrete table models live in models/.

Shared infrastructure lives in kit/. Run bin/server-check and bin/integration-test from the example root. No topic adds a separate dependency project or lockfile.

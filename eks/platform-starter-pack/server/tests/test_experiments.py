import asyncio

import httpx
from sqlmodel import select

from app.agents.execution import dispatch_once, finish_error, run_agent
from app.api import create_app
from app.kit.db.postgres import transaction
from app.models import Run, Task
from app.settings import Settings
from tests.test_auth import SECRET, token
from tests.test_integration import alembic


class MemoryObjects:
    def __init__(self, *args):
        self.values = {}

    async def put(self, key, content, content_type="text/plain"):
        import hashlib

        self.values[key] = content
        return len(content.encode()), hashlib.sha256(content.encode()).hexdigest()

    async def get(self, key):
        return self.values[key]


async def test_project_conversation_and_ownership(database, monkeypatch):
    await alembic("upgrade", "head")
    settings = Settings(env="test", database_url=database, token_secret=SECRET)
    app = create_app(settings)
    objects = MemoryObjects()
    monkeypatch.setattr("app.agents.execution.Objects", lambda _: objects)
    created = []

    class FakeWorkflows:
        def __init__(self, settings):
            pass

        async def get(self, name):
            return None

        async def create(self, name, run_id):
            created.append(run_id)

    monkeypatch.setattr("app.agents.execution.Workflows", FakeWorkflows)
    async with app.router.lifespan_context(app):
        app.state.tasks.objects = objects
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + token(scope="platform:read platform:verify")},
        ) as client:
            project = (
                await client.post("/api/projects", json={"name": "Storage experiment"})
            ).json()
            assert project["id"].startswith("prj_")
            assert project["updated_at"] and project["deleted_at"] is None
            response = await client.post(
                f"/api/projects/{project['id']}/tasks",
                json={
                    "title": "Inspect",
                    "prompt": "Inspect platform health",
                    "agent": "inspector",
                },
            )
            assert response.status_code == 201, response.text
            task_id = response.json()["id"]
            detail = (await client.get(f"/api/tasks/{task_id}")).json()
            assert detail["messages"][0]["content"] == "Inspect platform health"
            assert len(objects.values) == 1
            first_run = detail["runs"][0]["id"]
            followup = await client.post(
                f"/api/tasks/{task_id}/messages", json={"content": "Check again"}
            )
            assert followup.status_code == 201
            second_run = followup.json()["id"]
            # Multiple dispatchers cannot claim two runs of one task simultaneously.
            await asyncio.gather(
                dispatch_once(app.state.engine, settings), dispatch_once(app.state.engine, settings)
            )
            async with transaction(app.state.engine) as session:
                runs = (await session.exec(select(Run).where(Run.task_id == task_id))).all()
                assert sum(r.status == "running" for r in runs) == 1
            await run_agent(app.state.engine, settings, first_run)
            await dispatch_once(app.state.engine, settings)
            await run_agent(app.state.engine, settings, second_run)
            detail = (await client.get(f"/api/tasks/{task_id}")).json()
            assert [r["status"] for r in detail["runs"]] == ["completed", "completed"]
            assert len(detail["artifacts"]) == 2
            assert [m["role"] for m in detail["messages"]].count("assistant") == 2
            assert "PostgreSQL" in detail["messages"][-1]["content"]
            followup = await client.post(
                f"/api/tasks/{task_id}/messages", json={"content": "Cancel this run"}
            )
            cancelled_run = followup.json()["id"]
            await dispatch_once(app.state.engine, settings)
            assert (await client.post(f"/api/tasks/{task_id}/cancel")).status_code == 204
            # A late workflow failure or worker retry cannot undo cancellation.
            await finish_error(app.state.engine, cancelled_run, "late_failure")
            await run_agent(app.state.engine, settings, cancelled_run)
            detail = (await client.get(f"/api/tasks/{task_id}")).json()
            assert detail["runs"][-1]["status"] == "cancelled"
            assert len(detail["artifacts"]) == 2
            async with transaction(app.state.engine) as session:
                task = await session.get(Task, task_id)
                assert task is not None
                assert task.history_key in objects.values
                assert "history" not in Task.model_fields
                assert "prompt" not in Run.model_fields
            client.headers["Authorization"] = "Bearer " + token(
                sub="another-user", scope="platform:read platform:verify"
            )
            assert (await client.get(f"/api/tasks/{task_id}")).status_code == 404
            assert (await client.get("/api/projects")).json() == []
            client.headers["Authorization"] = "Bearer " + token(
                scope="platform:read platform:verify"
            )
            response = await client.delete(f"/api/projects/{project['id']}")
            assert response.status_code == 204
            assert (await client.get(f"/api/tasks/{task_id}")).status_code == 404
            assert (await client.get("/api/projects")).json() == []


async def test_object_failure_leaves_no_task(database):
    await alembic("upgrade", "head")
    app = create_app(Settings(env="test", database_url=database, token_secret=SECRET))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
            headers={"Authorization": "Bearer " + token(scope="platform:read platform:verify")},
        ) as client:
            project = (
                await client.post("/api/projects", json={"name": "Unavailable storage"})
            ).json()
            response = await client.post(
                f"/api/projects/{project['id']}/tasks",
                json={"title": "Will fail", "prompt": "Check"},
            )
            assert response.status_code == 500
            assert (await client.get(f"/api/projects/{project['id']}/tasks")).json() == []

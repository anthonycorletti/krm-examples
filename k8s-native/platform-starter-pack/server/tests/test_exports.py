import io
import json
from collections import Counter
from contextlib import asynccontextmanager

import pyarrow.parquet as pq
import pytest
from sqlmodel import select

from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.exports.execution import run_export, schedule_exports
from app.exports.service import ExportsService
from app.kit.db.postgres import create_db_engine, transaction
from app.kit.errors import Forbidden
from app.models import Export, Project, Task
from app.settings import Settings
from tests.test_integration import alembic


class MissingKey(Exception):
    pass


class MemoryObjects:
    class exceptions:
        NoSuchKey = MissingKey

    def __init__(self):
        self.values = {}

    @asynccontextmanager
    async def client(self):
        yield self

    async def put_object(self, Bucket, Key, Body, **kwargs):
        self.values[Key] = Body

    async def get_object(self, Bucket, Key):
        if Key not in self.values:
            raise MissingKey()
        data = self.values[Key]

        class Body:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self, size):
                return data[:size]

        return {"Body": Body()}

    async def put(self, key, body, content_type):
        self.values[key] = body.encode()


class MemoryWarehouse:
    fail = True
    rows = []

    async def request(self, query, body=b""):
        if self.fail:
            raise RuntimeError("Warehouse unavailable")
        if query.startswith("INSERT"):
            self.rows = pq.read_table(io.BytesIO(body)).to_pylist()

    async def query(self, query):
        return {
            "data": [
                {"entity": k, "count": n}
                for k, n in Counter(r["entity"] for r in self.rows).items()
            ]
        }


async def test_export_retry_ownership_soft_delete_and_schedule(database, monkeypatch):
    await alembic("upgrade", "head")
    settings = Settings(env="test", database_url=database)
    engine = create_db_engine(settings)
    service = ExportsService(engine, AuthService(settings))
    owner = Principal(
        subject="owner", environment="test", scopes=["platform:read", "platform:verify"]
    )
    viewer = Principal(subject="viewer", environment="test", scopes=["platform:read"])
    objects, warehouse = MemoryObjects(), MemoryWarehouse()
    monkeypatch.setattr("app.exports.execution.Objects", lambda _: objects)
    monkeypatch.setattr("app.exports.execution.Warehouse", lambda _: warehouse)
    try:
        async with transaction(engine) as session:
            project = Project(owner="owner", environment="test", name="Initial")
            session.add(project)
            session.add(Project(owner="someone-else", environment="test", name="Private"))
            await session.flush()
            session.add(
                Task(
                    project_id=project.id,
                    title="Task",
                    agent="inspector",
                    deleted_at=project.created_at,
                )
            )
        with pytest.raises(Forbidden):
            await service.create_export(viewer, "not-allowed")
        record = await service.create_export(owner, "one")
        assert (await service.create_export(owner, "one")).id == record.id
        with pytest.raises(RuntimeError, match="Warehouse unavailable"):
            await run_export(engine, settings, record.id)
        async with transaction(engine) as session:
            row = await session.get(Project, project.id)
            assert row is not None
            row.name = "Changed after export"
        warehouse.fail = False
        await run_export(engine, settings, record.id)
        assert len(warehouse.rows) == 2
        metadata = [json.loads(row["metadata_json"]) for row in warehouse.rows]
        assert any(row.get("name") == "Initial" for row in metadata)
        assert any(row.get("deleted_at") for row in metadata)
        assert not any(row.get("name") == "Private" for row in metadata)
        assert (await service.list_exports(owner))[0].status == "completed"
        assert await service.list_exports(viewer) == []
        await run_export(engine, settings, record.id)
        assert len(warehouse.rows) == 2
        await schedule_exports(engine, settings, "nightly-workflow-1")
        await schedule_exports(engine, settings, "nightly-workflow-1")
        async with transaction(engine) as session:
            scheduled = (
                await session.exec(select(Export).where(Export.trigger == "nightly"))
            ).all()
            assert len(scheduled) == 2
    finally:
        await engine.dispose()

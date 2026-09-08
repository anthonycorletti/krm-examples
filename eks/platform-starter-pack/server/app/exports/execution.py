"""Consistent, bounded metadata snapshots; only validated snapshots become current."""

import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select as core_select
from sqlalchemy import text
from sqlmodel import SQLModel, col, select

from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.exports.service import ExportsService, table_name
from app.kit.db.postgres import transaction
from app.kit.objects import Objects
from app.kit.warehouse import Warehouse
from app.kit.workflows import Workflows
from app.models import Artifact, Event, Export, Message, Project, Run, Task

ENTITIES = (Project, Task, Run, Message, Event, Artifact)
LIMIT = 100_000
MAX_BYTES = 64 * 1024 * 1024


def parquet(rows):
    schema = pa.schema(
        [(name, pa.string()) for name in ("entity", "entity_id", "project_id", "metadata_json")]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    body = sink.getvalue().to_pybytes()
    if len(body) > MAX_BYTES:
        raise ValueError("Export exceeds the 64 MiB demo limit")
    return body


def encode(value):
    return value.isoformat() if isinstance(value, datetime) else str(value)


async def capture(connection, record):
    rows = []
    await connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
    for model in ENTITIES:
        table = SQLModel.metadata.tables[model.__name__.lower()]
        query = core_select(table)
        if model is not Project:
            if model is not Task:
                query = query.join(Task, col(Task.id) == table.c.task_id)
            query = query.join(Project, col(Project.id) == col(Task.project_id))
        query = query.where(
            col(Project.owner) == record.owner, col(Project.environment) == record.environment
        )
        result = await connection.execute(query.limit(LIMIT + 1 - len(rows)))
        for row in result.mappings():
            value = dict(row)
            rows.append(
                {
                    "entity": table.name,
                    "entity_id": value["id"],
                    "project_id": value["id"] if model is Project else value.get("project_id", ""),
                    "metadata_json": json.dumps(value, default=encode, sort_keys=True),
                }
            )
        if len(rows) > LIMIT:
            raise ValueError("Export exceeds the 100,000-row demo limit")
    return rows


async def run_export(engine, settings, export_id):
    objects, warehouse = Objects(settings), Warehouse(settings)
    async with engine.connect() as lock:
        acquired = await lock.scalar(
            text("SELECT pg_try_advisory_lock(hashtext(:id))"), {"id": export_id}
        )
        await lock.commit()
        if not acquired:
            raise RuntimeError("Export is already running")
        try:
            async with transaction(engine) as session:
                record = await session.get(Export, export_id)
                if record is None:
                    raise RuntimeError("Unknown export")
                if record.status == "completed":
                    return
                record.status, record.error = "running", None
            prefix = f"{record.environment}/exports/{record.id}"
            manifest_key = f"{prefix}/manifest.json"
            # A committed manifest is immutable input for every retry of this export.
            async with objects.client() as client:
                try:
                    result = await client.get_object(
                        Bucket=settings.object_bucket, Key=manifest_key
                    )
                    async with result["Body"] as stream:
                        manifest = json.loads(await stream.read(1_048_576))
                except client.exceptions.NoSuchKey:
                    manifest = None
            if manifest is None:
                async with lock.begin():
                    rows = await capture(lock, record)
                body = await asyncio.to_thread(parquet, rows)
                counts = dict(Counter(row["entity"] for row in rows))
                key = f"{prefix}/metadata.parquet"
                manifest = {
                    "version": 1,
                    "export_id": record.id,
                    "owner": record.owner,
                    "environment": record.environment,
                    "row_counts": counts,
                    "key": key,
                    "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
                async with objects.client() as client:
                    await client.put_object(
                        Bucket=settings.object_bucket,
                        Key=key,
                        Body=body,
                        ContentType="application/vnd.apache.parquet",
                    )
                await objects.put(manifest_key, json.dumps(manifest), "application/json")
            async with objects.client() as client:
                result = await client.get_object(Bucket=settings.object_bucket, Key=manifest["key"])
                async with result["Body"] as stream:
                    body = await stream.read(MAX_BYTES + 1)
            if (
                len(body) != manifest["size"]
                or hashlib.sha256(body).hexdigest() != manifest["sha256"]
            ):
                raise ValueError("Export object checksum mismatch")
            table = "platform." + table_name(record.id)
            # This table is invisible to the API until SQL status is completed.
            await warehouse.request(f"DROP TABLE IF EXISTS {table}")
            await warehouse.request(
                f"CREATE TABLE {table} (entity String, entity_id String, "
                "project_id String, metadata_json String) "
                "ENGINE=MergeTree ORDER BY (entity, entity_id)"
            )
            await warehouse.request(f"INSERT INTO {table} FORMAT Parquet", body)
            result = await warehouse.query(
                f"SELECT entity, count() AS count FROM {table} GROUP BY entity FORMAT JSON"
            )
            actual = {row["entity"]: int(row["count"]) for row in result["data"]}
            if actual != manifest["row_counts"]:
                raise ValueError("Warehouse row counts do not match the export")
            async with transaction(engine) as session:
                current = await session.get(Export, export_id)
                assert current is not None
                current.status, current.manifest_key = "completed", manifest_key
                current.row_counts, current.error = actual, None
        except Exception:
            async with transaction(engine) as session:
                record = await session.get(Export, export_id)
                if record and record.status != "completed":
                    record.status, record.error = "failed", "Export failed; see workflow logs"
            raise
        finally:
            await lock.execute(text("SELECT pg_advisory_unlock(hashtext(:id))"), {"id": export_id})
            await lock.commit()


async def schedule_exports(engine, settings, request_key):
    async with transaction(engine) as session:
        owners = (
            await session.exec(
                select(Project.owner).where(Project.environment == settings.env).distinct()
            )
        ).all()
    service = ExportsService(engine, AuthService(settings))
    for owner in owners:
        principal = Principal(
            subject=owner, environment=settings.env, scopes=["platform:read", "platform:verify"]
        )
        await service.create_export(principal, request_key, trigger="nightly")


async def dispatch_exports(engine, settings):
    workflows = Workflows(settings)
    async with transaction(engine) as session:
        record = (
            await session.exec(
                select(Export)
                .where(Export.status == "queued")
                .order_by(Export.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
        ).first()
        if record:
            record.status = "running"
            record.workflow_name = "export-" + record.id.lower().replace("_", "-")
        active = list((await session.exec(select(Export).where(Export.status == "running"))).all())
    for record in active:
        workflow = await workflows.get(record.workflow_name)
        if workflow is None:
            await workflows.create(record.workflow_name, record.id, template="platform-export")
        elif workflow.get("status", {}).get("phase") in ("Failed", "Error", "Succeeded"):
            async with transaction(engine) as session:
                current = await session.get(Export, record.id)
                if current and current.status == "running":
                    current.status, current.error = (
                        "failed",
                        "Workflow ended without a validated snapshot",
                    )

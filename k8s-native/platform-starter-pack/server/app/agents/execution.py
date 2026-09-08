import asyncio
import json
import logging

from opentelemetry import trace
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits
from sqlmodel import col, select
from ulid import ULID

from app.agents.service import create_inspector
from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.components.service import ComponentsService
from app.kit.cache import Cache
from app.kit.db.postgres import transaction
from app.kit.objects import Objects
from app.kit.workflows import Workflows
from app.models import Artifact, Event, Message, Project, Run, Task

log = logging.getLogger(__name__)


async def finish_error(engine, run_id, reason):
    async with transaction(engine) as session:
        run = await session.get(Run, run_id)
        if run is None or run.status != "running":
            return
        task = (
            await session.exec(select(Task).where(Task.id == run.task_id).with_for_update())
        ).one()
        # Refresh after waiting for the task lock: cancellation may have won.
        run = (
            await session.exec(
                select(Run)
                .where(Run.id == run_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).one()
        if run.status != "running" or task.deleted_at:
            return
        run.status, run.error = "failed", reason
        if task and not task.deleted_at:
            task.status = "failed"
        session.add(Event(task_id=run.task_id, run_id=run.id, kind=reason))


async def run_agent(engine, settings, run_id):
    objects, cache = Objects(settings), Cache(settings)
    async with transaction(engine) as session:
        run = await session.get(Run, run_id)
        if run is None or run.status != "running":
            await cache.close()
            return
        task = await session.get(Task, run.task_id)
        project = await session.get(Project, task.project_id) if task else None
        if task is None or project is None or task.deleted_at or project.deleted_at:
            await cache.close()
            return
        session.add(Event(task_id=task.id, run_id=run.id, kind="agent_started"))
    principal = Principal(
        subject=project.owner, environment=project.environment, scopes=["platform:read"]
    )
    components = ComponentsService(engine, AuthService(settings))

    async def execute():
        history = (
            ModelMessagesTypeAdapter.validate_json(await objects.get(task.history_key))
            if task.history_key
            else []
        )
        prompt = await objects.get(run.prompt_key)
        if task.agent == "inspector":
            agent = create_inspector(TestModel(), components)
        else:
            if not settings.agent_model:
                raise RuntimeError("Model provider is not configured")
            agent = create_inspector(settings.agent_model, components)
        agent.instrument = InstrumentationSettings(include_content=False)
        result = await agent.run(
            prompt,
            deps=principal,
            message_history=history,
            usage_limits=UsageLimits(request_limit=12, total_tokens_limit=24000),
        )
        prefix = f"{project.environment}/{project.id}/{task.id}/{run.id}/{ULID()}"
        history_key, report_key = f"{prefix}/history.json", f"{prefix}/report.txt"
        await objects.put(history_key, result.all_messages_json().decode(), "application/json")
        output = result.output
        if task.agent == "inspector":
            # The deterministic tool result is evidence, not a simulated model answer.
            try:
                results = json.loads(output)["inspect_platform"]
                output = "\n\n".join(
                    f"{item['name']} — {item['status']}\n{item['detail']}" for item in results
                )
            except (ValueError, KeyError, TypeError):
                pass
            output = "Deterministic platform inspection (no language model).\n\n" + output
        size, digest = await objects.put(report_key, output)
        span_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
        async with transaction(engine) as session:
            current_task = (
                await session.exec(select(Task).where(Task.id == task.id).with_for_update())
            ).one()
            current = (
                await session.exec(select(Run).where(Run.id == run.id).with_for_update())
            ).one()
            if current.status != "running" or current_task.deleted_at:
                return
            current.status, current.trace_id = "completed", span_id
            current.input_tokens = result.usage.input_tokens
            current.output_tokens = result.usage.output_tokens
            current_task.history_key, current_task.status = history_key, "completed"
            session.add(
                Message(
                    task_id=task.id,
                    run_id=run.id,
                    role="assistant",
                    object_key=report_key,
                    size=size,
                    sha256=digest,
                )
            )
            session.add(
                Artifact(
                    task_id=task.id,
                    run_id=run.id,
                    name="report.txt",
                    object_key=report_key,
                    size=size,
                    sha256=digest,
                )
            )
            session.add(Event(task_id=task.id, run_id=run.id, kind="completed"))
        await cache.progress(run.id, "completed")

    async def monitor(execution):
        while not execution.done():
            await asyncio.sleep(1)
            async with transaction(engine) as session:
                current = await session.get(Run, run.id)
                if current is None or current.status != "running":
                    if current is None or current.status == "cancelled":
                        execution.cancel()
                    return
            await cache.progress(run.id, "running")

    execution = None
    watcher = None
    try:
        with trace.get_tracer("platform.agent").start_as_current_span(
            "agent.run", attributes={"run.id": run.id, "task.id": task.id, "agent.name": task.agent}
        ):
            async with asyncio.timeout(300):
                execution = asyncio.create_task(execute())
                watcher = asyncio.create_task(monitor(execution))
                await execution
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Agent run failed: %s", run.id)
        await finish_error(engine, run.id, "agent_failed")
        raise
    finally:
        for pending in (execution, watcher):
            if pending and not pending.done():
                pending.cancel()
        await asyncio.gather(*(p for p in (execution, watcher) if p), return_exceptions=True)
        await cache.close()


async def dispatch_once(engine, settings):
    workflows = Workflows(settings)
    async with transaction(engine) as session:
        task = (
            await session.exec(
                select(Task)
                .join(Project, col(Project.id) == col(Task.project_id))
                .where(
                    select(Run.id).where(Run.task_id == Task.id, Run.status == "queued").exists(),
                    Task.status != "running",
                    col(Task.deleted_at).is_(None),
                    col(Project.deleted_at).is_(None),
                )
                .order_by(Task.id)
                .with_for_update(of=Task, skip_locked=True)
                .limit(1)
            )
        ).first()
        if task:
            run = (
                await session.exec(
                    select(Run)
                    .where(Run.task_id == task.id, Run.status == "queued")
                    .order_by(Run.id)
                    .with_for_update()
                    .limit(1)
                )
            ).one()
            run.status, task.status = "running", "running"
            run.workflow_name = f"agent-{run.id.lower().replace('_', '-')}"
            session.add(Event(task_id=task.id, run_id=run.id, kind="workflow_submitted"))
    async with transaction(engine) as session:
        active = list(
            (
                await session.exec(
                    select(Run)
                    .where(
                        col(Run.status).in_(("running", "cancelled")),
                        col(Run.workflow_name).is_not(None),
                    )
                    .order_by(col(Run.created_at).desc())
                    .limit(100)
                )
            ).all()
        )
    for run in active:
        workflow = await workflows.get(run.workflow_name)
        if run.status == "cancelled":
            if workflow and workflow.get("status", {}).get("phase") not in (
                "Succeeded",
                "Failed",
                "Error",
            ):
                await workflows.cancel(run.workflow_name)
        elif workflow is None:
            await workflows.create(run.workflow_name, run.id)
        elif workflow.get("status", {}).get("phase") in ("Failed", "Error", "Succeeded"):
            await finish_error(engine, run.id, "workflow_ended_without_result")


async def dispatch(engine, settings):
    await Objects(settings).ensure_bucket()
    while True:
        try:
            await dispatch_once(engine, settings)
            from app.exports.execution import dispatch_exports

            await dispatch_exports(engine, settings)
        except Exception:
            log.exception("Workflow reconciliation failed; queued work is retained")
        await asyncio.sleep(3)

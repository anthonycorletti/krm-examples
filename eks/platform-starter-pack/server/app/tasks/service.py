from fastapi import HTTPException
from sqlmodel import col, select
from ulid import ULID

from app.auth.schemas import Principal
from app.kit.cache import Cache
from app.kit.db.postgres import transaction
from app.kit.objects import Objects
from app.kit.utils import utc_now
from app.models import Artifact, Event, Message, Run, Task
from app.projects.service import ProjectsService
from app.tasks.schemas import AgentRead, MessageRead, TaskCreate, TaskDetail


class TasksService(ProjectsService):
    def __init__(self, engine, auth, objects: Objects):
        super().__init__(engine, auth)
        self.objects = objects

    def agents(self) -> list[AgentRead]:
        model = self.auth.settings.agent_model
        return [
            AgentRead(
                id="inspector",
                name="Platform inspector",
                model="TestModel",
                description="Real platform tools with a deterministic model.",
                available=True,
            ),
            AgentRead(
                id="researcher",
                name="Research agent",
                model=model or "Not configured",
                description="Investigate a task and report on platform evidence.",
                available=bool(model),
            ),
            AgentRead(
                id="analyst",
                name="Analysis agent",
                model=model or "Not configured",
                description="Analyze evidence, compare results, and propose next experiments.",
                available=bool(model),
            ),
        ]

    async def owned_task(self, session, principal, task_id: str, lock=False) -> Task:
        task = await session.get(Task, task_id)
        if task is None or task.deleted_at:
            raise HTTPException(404, "Task not found")
        await self.owned(session, principal, task.project_id, lock=lock)
        if lock:
            task = (
                await session.exec(
                    select(Task)
                    .where(Task.id == task_id, col(Task.deleted_at).is_(None))
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            ).first()
            if task is None:
                raise HTTPException(404, "Task not found")
        return task

    async def list_tasks(self, principal: Principal, project_id: str) -> list[Task]:
        async with transaction(self.engine) as session:
            await self.owned(session, principal, project_id)
            return list(
                (
                    await session.exec(
                        select(Task)
                        .where(Task.project_id == project_id, col(Task.deleted_at).is_(None))
                        .order_by(col(Task.created_at).desc())
                        .limit(100)
                    )
                ).all()
            )

    async def create_task(self, principal: Principal, project_id: str, data: TaskCreate) -> Task:
        self.permission(principal, True)
        if not any(a.id == data.agent and a.available for a in self.agents()):
            raise HTTPException(422, "Agent is not configured")
        async with transaction(self.engine) as session:
            await self.owned(session, principal, project_id)
        task = Task(project_id=project_id, title=data.title.strip(), agent=data.agent)
        if not task.title or not data.prompt.strip():
            raise HTTPException(422, "Title and instructions are required")
        run, message = await self.prepare_message(principal, task, data.prompt)
        async with transaction(self.engine) as session:
            await self.owned(session, principal, project_id, lock=True)
            session.add(task)
            await session.flush()
            session.add(run)
            await session.flush()
            session.add(message)
            session.add(Event(task_id=task.id, run_id=run.id, kind="queued"))
        return task

    async def prepare_message(self, principal, task, content):
        key = f"{principal.environment}/{task.project_id}/{task.id}/{ULID()}.txt"
        size, digest = await self.objects.put(key, content)
        run = Run(task_id=task.id, prompt_key=key)
        return run, Message(
            task_id=task.id, run_id=run.id, role="user", object_key=key, size=size, sha256=digest
        )

    async def send(self, principal: Principal, task_id: str, content: str) -> Run:
        self.permission(principal, True)
        if not content.strip():
            raise HTTPException(422, "Instructions are required")
        async with transaction(self.engine) as session:
            task = await self.owned_task(session, principal, task_id)
        run, message = await self.prepare_message(principal, task, content)
        async with transaction(self.engine) as session:
            task = await self.owned_task(session, principal, task_id, lock=True)
            if task.status != "running":
                task.status = "queued"
            session.add(run)
            await session.flush()
            session.add(message)
            session.add(Event(task_id=task.id, run_id=run.id, kind="queued"))
        return run

    async def detail(self, principal: Principal, task_id: str) -> TaskDetail:
        async with transaction(self.engine) as session:
            task = await self.owned_task(session, principal, task_id)
            rows = {}
            for name, model in (
                ("messages", Message),
                ("runs", Run),
                ("events", Event),
                ("artifacts", Artifact),
            ):
                rows[name] = list(
                    (
                        await session.exec(
                            select(model)
                            .where(model.task_id == task_id, col(model.deleted_at).is_(None))
                            .order_by(col(model.created_at).desc(), col(model.id).desc())
                            .limit(200)
                        )
                    ).all()
                )
                rows[name].reverse()
        messages = []
        for message in rows["messages"]:
            assert isinstance(message, Message)
            content = await self.objects.get(message.object_key)
            messages.append(MessageRead(**message.model_dump(), content=content))
        cache = Cache(self.auth.settings)
        progress = {}
        try:
            for run in rows["runs"]:
                value = await cache.read_progress(run.id)
                if value:
                    progress[run.id] = value
        finally:
            await cache.close()
        return TaskDetail(
            task=task,
            messages=messages,
            runs=rows["runs"],
            events=rows["events"],
            artifacts=rows["artifacts"],
            live_progress=progress,
        )

    async def cancel(self, principal: Principal, task_id: str, delete: bool = False):
        self.permission(principal, True)
        async with transaction(self.engine) as session:
            task = await self.owned_task(session, principal, task_id, lock=True)
            task.status = "cancelled"
            if delete:
                task.deleted_at = utc_now()
            for run in (
                await session.exec(
                    select(Run).where(
                        Run.task_id == task_id, col(Run.status).in_(("queued", "running"))
                    )
                )
            ).all():
                run.status = "cancelled"
                session.add(Event(task_id=task_id, run_id=run.id, kind="cancelled"))

    async def artifact(self, principal, task_id, artifact_id):
        async with transaction(self.engine) as session:
            await self.owned_task(session, principal, task_id)
            artifact = await session.get(Artifact, artifact_id)
            if artifact is None or artifact.task_id != task_id or artifact.deleted_at:
                raise HTTPException(404, "Artifact not found")
        return await self.objects.get(artifact.object_key)

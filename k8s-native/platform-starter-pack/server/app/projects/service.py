from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select

from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.kit.db.postgres import transaction
from app.kit.utils import utc_now
from app.models import Project, Run, Task
from app.projects.schemas import ProjectCreate


class ProjectsService:
    def __init__(self, engine: AsyncEngine, auth: AuthService):
        self.engine, self.auth = engine, auth

    def permission(self, principal: Principal, write: bool = False):
        self.auth.authorize(
            principal, "platform:verify" if write else "platform:read", self.auth.settings.env
        )

    async def owned(self, session, principal: Principal, project_id: str, lock=False) -> Project:
        self.permission(principal)
        query = select(Project).where(
            Project.id == project_id,
            Project.owner == principal.subject,
            Project.environment == principal.environment,
            col(Project.deleted_at).is_(None),
        )
        if lock:
            query = query.with_for_update()
        project = (await session.exec(query)).first()
        if project is None:
            raise HTTPException(404, "Project not found")
        return project

    async def list(self, principal: Principal) -> list[Project]:
        self.permission(principal)
        async with transaction(self.engine) as session:
            return list(
                (
                    await session.exec(
                        select(Project)
                        .where(
                            Project.owner == principal.subject,
                            Project.environment == principal.environment,
                            col(Project.deleted_at).is_(None),
                        )
                        .order_by(col(Project.created_at).desc())
                        .limit(100)
                    )
                ).all()
            )

    async def create(self, principal: Principal, data: ProjectCreate) -> Project:
        self.permission(principal, True)
        if not data.name.strip():
            raise HTTPException(422, "Project name is required")
        project = Project(
            name=data.name.strip(),
            description=data.description,
            owner=principal.subject,
            environment=principal.environment,
        )
        async with transaction(self.engine) as session:
            session.add(project)
        return project

    async def delete(self, principal: Principal, project_id: str):
        self.permission(principal, True)
        async with transaction(self.engine) as session:
            project = await self.owned(session, principal, project_id, lock=True)
            now = utc_now()
            project.deleted_at = now
            for task in (
                await session.exec(
                    select(Task)
                    .where(Task.project_id == project_id)
                    .order_by(Task.id)
                    .with_for_update()
                )
            ).all():
                task.deleted_at, task.status = now, "cancelled"
                for run in (await session.exec(select(Run).where(Run.task_id == task.id))).all():
                    run.deleted_at = now
                    if run.status in ("queued", "running"):
                        run.status = "cancelled"

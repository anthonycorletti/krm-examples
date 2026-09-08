from fastapi import APIRouter, Depends, Request, Response

from app.auth.router import current_principal
from app.auth.schemas import Principal
from app.models import Project
from app.projects.schemas import ProjectCreate

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", operation_id="list_projects")
async def list_projects(
    request: Request, principal: Principal = Depends(current_principal)
) -> list[Project]:
    return await request.app.state.projects.list(principal)


@router.post("", status_code=201, operation_id="create_project")
async def create_project(
    data: ProjectCreate, request: Request, principal: Principal = Depends(current_principal)
) -> Project:
    return await request.app.state.projects.create(principal, data)


@router.delete("/{project_id}", status_code=204, operation_id="delete_project")
async def delete_project(
    project_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> Response:
    await request.app.state.projects.delete(principal, project_id)
    return Response(status_code=204)

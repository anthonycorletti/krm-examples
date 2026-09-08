from fastapi import APIRouter, Depends, Request, Response

from app.auth.router import current_principal
from app.auth.schemas import Principal
from app.models import Run, Task
from app.tasks.schemas import AgentRead, MessageCreate, TaskCreate, TaskDetail

router = APIRouter(prefix="/api", tags=["tasks"])


@router.get("/agents", operation_id="list_agents")
async def agents(
    request: Request, principal: Principal = Depends(current_principal)
) -> list[AgentRead]:
    request.app.state.tasks.permission(principal)
    return request.app.state.tasks.agents()


@router.get("/projects/{project_id}/tasks", operation_id="list_tasks")
async def tasks(
    project_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> list[Task]:
    return await request.app.state.tasks.list_tasks(principal, project_id)


@router.post("/projects/{project_id}/tasks", status_code=201, operation_id="create_task")
async def create(
    project_id: str,
    data: TaskCreate,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> Task:
    return await request.app.state.tasks.create_task(principal, project_id, data)


@router.get("/tasks/{task_id}", operation_id="read_task")
async def read(
    task_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> TaskDetail:
    return await request.app.state.tasks.detail(principal, task_id)


@router.post("/tasks/{task_id}/messages", status_code=201, operation_id="send_message")
async def send(
    task_id: str,
    data: MessageCreate,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> Run:
    return await request.app.state.tasks.send(principal, task_id, data.content)


@router.post("/tasks/{task_id}/cancel", status_code=204, operation_id="cancel_task")
async def cancel(
    task_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> Response:
    await request.app.state.tasks.cancel(principal, task_id)
    return Response(status_code=204)


@router.delete("/tasks/{task_id}", status_code=204, operation_id="delete_task")
async def delete(
    task_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> Response:
    await request.app.state.tasks.cancel(principal, task_id, delete=True)
    return Response(status_code=204)


@router.get("/tasks/{task_id}/artifacts/{artifact_id}", operation_id="read_artifact")
async def artifact(
    task_id: str,
    artifact_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> str:
    return await request.app.state.tasks.artifact(principal, task_id, artifact_id)

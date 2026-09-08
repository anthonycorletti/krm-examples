from fastapi import APIRouter, Depends, Request

from app.auth.router import current_principal
from app.auth.schemas import Principal
from app.components.schemas import ComponentRead

router = APIRouter(prefix="/api/components", tags=["components"])


@router.get("", operation_id="list_components")
async def list_components(
    request: Request,
    environment: str = "local",
    principal: Principal = Depends(current_principal),
) -> list[ComponentRead]:
    return await request.app.state.components.list(principal, environment)

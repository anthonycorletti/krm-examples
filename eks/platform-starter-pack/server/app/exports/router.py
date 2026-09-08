from fastapi import APIRouter, Depends, Request

from app.auth.router import current_principal
from app.auth.schemas import Principal
from app.exports.schemas import ExportCreate, WarehouseRead
from app.models import Export

router = APIRouter(prefix="/api/exports", tags=["exports"])


@router.get("", operation_id="list_exports")
async def list_exports(
    request: Request, principal: Principal = Depends(current_principal)
) -> list[Export]:
    return await request.app.state.exports.list_exports(principal)


@router.post("", status_code=202, operation_id="create_export")
async def create_export(
    data: ExportCreate, request: Request, principal: Principal = Depends(current_principal)
) -> Export:
    return await request.app.state.exports.create_export(principal, data.request_key)


@router.get("/warehouse", operation_id="read_warehouse")
async def warehouse(
    request: Request, principal: Principal = Depends(current_principal)
) -> WarehouseRead:
    return await request.app.state.exports.warehouse(principal)


@router.get("/{export_id}/manifest", operation_id="read_export_manifest")
async def manifest(
    export_id: str, request: Request, principal: Principal = Depends(current_principal)
) -> dict:
    return await request.app.state.exports.manifest(principal, export_id)

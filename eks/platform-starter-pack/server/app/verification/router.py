from fastapi import APIRouter, Depends, Request

from app.auth.router import current_principal
from app.auth.schemas import Principal
from app.verification.schemas import MigrationRead, VerificationCreate, VerificationRead

router = APIRouter(prefix="/api", tags=["verification"])


@router.post("/verifications", operation_id="run_verification", status_code=201)
async def run_verification(
    body: VerificationCreate,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> VerificationRead:
    return await request.app.state.verification.run(principal, body.environment)


@router.get("/verifications", operation_id="list_verifications")
async def list_verifications(
    request: Request,
    environment: str = "local",
    principal: Principal = Depends(current_principal),
) -> list[VerificationRead]:
    return await request.app.state.verification.history(principal, environment)


@router.get("/migrations", operation_id="read_migrations")
async def read_migrations(
    request: Request,
    environment: str = "local",
    principal: Principal = Depends(current_principal),
) -> MigrationRead:
    return await request.app.state.verification.migrations(principal, environment)

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.verification.service import expected_revision

router = APIRouter(tags=["health"])


@router.get("/livez", operation_id="read_liveness")
async def livez() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/readyz", operation_id="read_readiness")
async def readyz(request: Request) -> dict[str, str]:
    try:
        async with request.app.state.engine.connect() as connection:
            revision = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar()
        if revision != expected_revision():
            raise HTTPException(503, "Database migration required")
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable or not migrated") from exc
    return {"status": "ready"}

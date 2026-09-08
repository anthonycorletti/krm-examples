import hashlib
import json
import re

from fastapi import HTTPException
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import col, select

from app.exports.schemas import WarehouseRead
from app.kit.db.postgres import transaction
from app.kit.objects import Objects
from app.kit.warehouse import Warehouse
from app.models import Export
from app.projects.service import ProjectsService


def table_name(export_id: str) -> str:
    if not re.fullmatch(r"exp_[0-9A-HJKMNP-TV-Z]{26}", export_id):
        raise ValueError("Invalid export ID")
    return "snapshot_" + export_id.lower()


class ExportsService(ProjectsService):
    async def create_export(self, principal, request_key, trigger="manual"):
        self.permission(principal, True)
        key = hashlib.sha256(
            f"{principal.environment}:{principal.subject}:{request_key}".encode()
        ).hexdigest()
        record = Export(
            owner=principal.subject,
            environment=principal.environment,
            request_key=key,
            trigger=trigger,
        )
        async with transaction(self.engine) as session:
            await session.exec(
                insert(Export)
                .values(**record.model_dump())
                .on_conflict_do_nothing(index_elements=["request_key"])
            )
            record = (await session.exec(select(Export).where(Export.request_key == key))).one()
        return record

    async def list_exports(self, principal):
        self.permission(principal)
        async with transaction(self.engine) as session:
            return list(
                (
                    await session.exec(
                        select(Export)
                        .where(
                            Export.owner == principal.subject,
                            Export.environment == principal.environment,
                            col(Export.deleted_at).is_(None),
                        )
                        .order_by(col(Export.created_at).desc())
                        .limit(30)
                    )
                ).all()
            )

    async def warehouse(self, principal):
        self.permission(principal)
        async with transaction(self.engine) as session:
            record = (
                await session.exec(
                    select(Export)
                    .where(
                        Export.owner == principal.subject,
                        Export.environment == principal.environment,
                        Export.status == "completed",
                        col(Export.deleted_at).is_(None),
                    )
                    .order_by(col(Export.created_at).desc())
                    .limit(1)
                )
            ).first()
        if record is None:
            return WarehouseRead(export=None, counts={})
        try:
            result = await Warehouse(self.auth.settings).query(
                f"SELECT entity, count() AS count FROM platform.{table_name(record.id)} "
                "GROUP BY entity FORMAT JSON"
            )
            counts = {r["entity"]: int(r["count"]) for r in result["data"]}
        except Exception as exc:
            raise HTTPException(
                503, "Warehouse query failed; previous export metadata is retained"
            ) from exc
        return WarehouseRead(export=record, counts=counts)

    async def manifest(self, principal, export_id):
        self.permission(principal)
        async with transaction(self.engine) as session:
            record = await session.get(Export, export_id)
            if (
                not record
                or record.owner != principal.subject
                or record.environment != principal.environment
                or record.deleted_at
            ):
                raise HTTPException(404, "Export not found")
            if not record.manifest_key:
                raise HTTPException(409, "Export is not validated yet")
        return json.loads(await Objects(self.auth.settings).get(record.manifest_key))

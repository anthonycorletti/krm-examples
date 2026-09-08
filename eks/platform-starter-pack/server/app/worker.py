"""Trusted, bounded agent task entrypoint for local demos and Argo Workflows."""

import argparse
import asyncio
import json

from pydantic_ai.models.test import TestModel

from app.agents.execution import dispatch, run_agent
from app.agents.service import create_inspector
from app.auth.schemas import Principal
from app.auth.service import AuthService
from app.components.service import ComponentsService
from app.kit.db.postgres import create_db_engine
from app.kit.telemetry import configure
from app.settings import Settings


async def inspect_demo(settings: Settings) -> dict[str, str]:
    engine = create_db_engine(settings)
    try:
        agent = create_inspector(TestModel(), ComponentsService(engine, AuthService(settings)))
        result = await agent.run(
            "Inspect the platform using the inspect_platform tool.",
            deps=Principal(
                subject="platform-inspector-demo",
                environment=settings.env,
                scopes=["platform:read"],
            ),
        )
        return {
            "environment": settings.env,
            "model": "pydantic-ai TestModel (deterministic demo; no model provider)",
            "output": result.output,
        }
    finally:
        await engine.dispose()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("demo", "dispatch", "run", "export", "schedule-exports"),
        default="demo",
        nargs="?",
    )
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    settings = Settings()
    if args.action == "demo":
        async with asyncio.timeout(60):
            print(json.dumps(await inspect_demo(settings)))
        return
    provider = configure(settings, "platform-worker")
    engine = create_db_engine(settings)
    try:
        if args.action == "dispatch":
            await dispatch(engine, settings)
        elif args.run_id and args.action == "export":
            from app.exports.execution import run_export

            await run_export(engine, settings, args.run_id)
        elif args.run_id and args.action == "schedule-exports":
            from app.exports.execution import schedule_exports

            await schedule_exports(engine, settings, args.run_id)
        elif args.run_id:
            await run_agent(engine, settings, args.run_id)
        else:
            parser.error("run requires a run ID")
    finally:
        await engine.dispose()
        if provider:
            await asyncio.to_thread(provider.shutdown)


if __name__ == "__main__":
    asyncio.run(main())

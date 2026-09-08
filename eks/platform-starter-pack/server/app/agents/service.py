from pydantic_ai import Agent, RunContext

from app.auth.schemas import Principal
from app.components.service import ComponentsService


def create_inspector(model, components: ComponentsService) -> Agent[Principal, str]:
    agent = Agent(
        model,
        deps_type=Principal,
        instructions="Inspect platform evidence. Never claim untested components work.",
    )

    @agent.tool
    async def inspect_platform(ctx: RunContext[Principal]):
        return await components.list(ctx.deps, ctx.deps.environment)

    return agent

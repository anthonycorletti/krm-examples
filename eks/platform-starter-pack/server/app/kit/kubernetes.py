"""Small async reader for explicitly authorized Kubernetes resource status."""

import asyncio
import ssl
from pathlib import Path

import httpx

ROOT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


async def get_resource(path: str) -> dict:
    token = await asyncio.to_thread((ROOT / "token").read_text)
    context = await asyncio.to_thread(ssl.create_default_context, cafile=str(ROOT / "ca.crt"))
    async with httpx.AsyncClient(
        base_url="https://kubernetes.default.svc",
        verify=context,
        timeout=3,
        headers={"Authorization": f"Bearer {token.strip()}"},
    ) as client:
        response = await client.get(path)
        response.raise_for_status()
        return response.json()


def condition(resource: dict, name: str) -> bool:
    generation = resource.get("metadata", {}).get("generation", 0)
    return any(
        c.get("type") == name
        and c.get("status") == "True"
        and c.get("observedGeneration", generation) >= generation
        for c in resource.get("status", {}).get("conditions", [])
    )

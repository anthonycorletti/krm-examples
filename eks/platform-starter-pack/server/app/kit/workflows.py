import ssl
from pathlib import Path

import httpx

from app.settings import Settings


class Workflows:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    async def request(self, method, path="", **kwargs):
        context = ssl.create_default_context(cafile=str(self.root / "ca.crt"))
        async with httpx.AsyncClient(
            base_url="https://kubernetes.default.svc",
            verify=context,
            timeout=10,
            headers={"Authorization": f"Bearer {(self.root / 'token').read_text().strip()}"},
        ) as client:
            return await client.request(
                method,
                f"/apis/argoproj.io/v1alpha1/namespaces/{self.settings.workflow_namespace}/workflows{path}",
                **kwargs,
            )

    async def get(self, name):
        response = await self.request("GET", f"/{name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def create(self, name, run_id):
        response = await self.request(
            "POST",
            json={
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {"name": name, "labels": {"app": "platform-worker"}},
                "spec": {
                    "workflowTemplateRef": {"name": "platform-agent"},
                    "arguments": {
                        "parameters": [
                            {"name": "run-id", "value": run_id},
                            {"name": "server-image", "value": self.settings.worker_image},
                        ]
                    },
                },
            },
        )
        if response.status_code != 409:
            response.raise_for_status()

    async def cancel(self, name):
        response = await self.request(
            "PATCH",
            f"/{name}",
            headers={"Content-Type": "application/merge-patch+json"},
            json={"spec": {"shutdown": "Terminate"}},
        )
        if response.status_code != 404:
            response.raise_for_status()

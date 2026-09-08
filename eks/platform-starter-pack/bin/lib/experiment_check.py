"""Exercise the local HTTPS project/task flow and leave inspectable demo evidence."""

import asyncio
import json
import ssl
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


async def main():
    process = await asyncio.create_subprocess_exec(
        str(ROOT / "bin/local-token"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError("Could not obtain the local sign-in token")
    context = ssl.create_default_context(cafile=str(ROOT / ".local/tls/ca.crt"))
    # The already-trusted development CA predates AKI/SKI extensions. Keep chain
    # and hostname verification, while accepting that legacy local certificate.
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    async with httpx.AsyncClient(
        base_url="https://127.0.0.1:8000",
        verify=context,
        headers={"Host": "api.localhost", "Authorization": "Bearer " + output.decode().strip()},
        timeout=45,
    ) as client:

        async def request(method, path, **kwargs):
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        if "--components" in sys.argv:
            components = await request("GET", "/api/components")
            print(json.dumps(components, indent=2))
            return

        project = await request(
            "POST",
            "/api/projects",
            json={
                "name": "Platform walkthrough",
                "description": "Postgres, object storage, Valkey, Argo, and telemetry.",
            },
        )
        task = await request(
            "POST",
            f"/api/projects/{project['id']}/tasks",
            json={
                "title": "Inspect the running platform",
                "prompt": "Inspect platform services. Distinguish live evidence from planned work.",
                "agent": "inspector",
            },
        )

        async def wait_runs(count):
            for _ in range(120):
                detail = await request("GET", f"/api/tasks/{task['id']}")
                runs = detail["runs"]
                if any(r["status"] in ("failed", "cancelled") for r in runs):
                    raise RuntimeError(
                        f"Task failed: {[(r['id'], r['status'], r['error']) for r in runs]}"
                    )
                if len(runs) == count and all(r["status"] == "completed" for r in runs):
                    return detail
                await asyncio.sleep(2)
            raise RuntimeError("Timed out waiting for the Argo agent task")

        await wait_runs(1)
        await request(
            "POST",
            f"/api/tasks/{task['id']}/messages",
            json={"content": "Inspect the services again and preserve the previous conversation."},
        )
        detail = await wait_runs(2)
        assert len(detail["messages"]) == 4, detail
        assert len(detail["artifacts"]) == 2
        assert all(
            r["workflow_name"] and r["trace_id"] and int(r["trace_id"], 16) for r in detail["runs"]
        )
        assert detail["live_progress"], "Valkey progress was not observed"
        report = await request(
            "GET", f"/api/tasks/{task['id']}/artifacts/{detail['artifacts'][0]['id']}"
        )
        assert "PostgreSQL" in report and "SeaweedFS" in report
        print(
            json.dumps(
                {
                    "project": project["id"],
                    "task": task["id"],
                    "runs": [r["id"] for r in detail["runs"]],
                    "url": f"https://web.localhost:5173/?project={project['id']}&task={task['id']}",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

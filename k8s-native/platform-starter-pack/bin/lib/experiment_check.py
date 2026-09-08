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

        if "--identity" in sys.argv:
            await request("GET", "/api/projects")
            process = await asyncio.create_subprocess_exec(
                str(ROOT / "bin/local-token"), "--mcp", stdout=asyncio.subprocess.PIPE
            )
            mcp_token, _ = await process.communicate()
            assert process.returncode == 0
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "platform-check", "version": "1"},
                },
            }
            async with httpx.AsyncClient(verify=context, timeout=45) as mcp:
                headers = {
                    "Host": "mcp.localhost",
                    "Accept": "application/json, text/event-stream",
                    "Authorization": "Bearer " + mcp_token.decode().strip(),
                }
                response = await mcp.post(
                    "https://127.0.0.1:8001/mcp", headers=headers, json=payload
                )
                response.raise_for_status()
                assert '"serverInfo"' in response.text
                headers["Authorization"] = client.headers["Authorization"]
                response = await mcp.post(
                    "https://127.0.0.1:8001/mcp", headers=headers, json=payload
                )
                assert response.status_code == 401, "MCP accepted an API audience token"
            response = await client.get(
                "/api/projects", headers={"Authorization": "Bearer " + mcp_token.decode().strip()}
            )
            assert response.status_code == 401, "API accepted an MCP audience token"
            records = await request("GET", "/api/exports")
            nightly = next(
                (r for r in records if r["trigger"] == "nightly" and r["status"] == "completed"),
                None,
            )
            print(
                json.dumps(
                    {
                        "api_oidc": True,
                        "mcp_oidc": True,
                        "audience_isolation": True,
                        "completed_nightly_export": nightly["id"] if nightly else None,
                    },
                    indent=2,
                )
            )
            return

        if "--export" in sys.argv:
            key = "local-check-" + str(__import__("time").time_ns())
            record = await request("POST", "/api/exports", json={"request_key": key})
            again = await request("POST", "/api/exports", json={"request_key": key})
            assert record["id"] == again["id"], "Idempotent submission changed IDs"
            for _ in range(180):
                records = await request("GET", "/api/exports")
                current = next(r for r in records if r["id"] == record["id"])
                if current["status"] == "completed":
                    break
                if current["status"] == "failed":
                    raise RuntimeError(f"Export failed: {current['workflow_name']}")
                await asyncio.sleep(2)
            else:
                raise RuntimeError("Export timed out")
            manifest = await request("GET", f"/api/exports/{record['id']}/manifest")
            warehouse = await request("GET", "/api/exports/warehouse")
            assert manifest["row_counts"] == warehouse["counts"]
            assert warehouse["export"]["id"] == record["id"]
            viewer_process = await asyncio.create_subprocess_exec(
                str(ROOT / "bin/local-token"), "--viewer", stdout=asyncio.subprocess.PIPE
            )
            viewer, _ = await viewer_process.communicate()
            assert viewer_process.returncode == 0
            client.headers["Authorization"] = "Bearer " + viewer.decode().strip()
            assert (await client.post("/api/exports", json={"request_key": key})).status_code == 403
            assert (await client.get(f"/api/exports/{record['id']}/manifest")).status_code == 404
            print(
                json.dumps(
                    {
                        "export": record["id"],
                        "workflow": current["workflow_name"],
                        "counts": warehouse["counts"],
                        "viewer_denied": True,
                    },
                    indent=2,
                )
            )
            return

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

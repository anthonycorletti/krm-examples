import httpx

from app.auth.service import AuthService
from app.components.schemas import ComponentRead
from app.components.service import ComponentsService
from app.kit.kubernetes import condition
from app.settings import Settings


def test_stale_condition_is_not_ready():
    resource = {
        "metadata": {"generation": 3},
        "status": {"conditions": [{"type": "Ready", "status": "True", "observedGeneration": 2}]},
    }
    assert not condition(resource, "Ready")
    resource["status"]["conditions"][0]["observedGeneration"] = 3
    assert condition(resource, "Ready")
    assert not condition(resource, "Available")


async def test_component_probes_distinguish_missing_permissions_and_readiness(monkeypatch):
    async def get(path):
        if "horizontalpodautoscalers" in path or "clusters/" in path:
            code = 404 if "horizontalpodautoscalers" in path else 403
            response = httpx.Response(code, request=httpx.Request("GET", "https://kubernetes.test"))
            response.raise_for_status()
        if "gateways" in path:
            return {
                "metadata": {"generation": 2},
                "status": {
                    "conditions": [
                        {"type": "Programmed", "status": "True", "observedGeneration": 1},
                        {"type": "Accepted", "status": "True", "observedGeneration": 2},
                    ]
                },
            }
        return {
            "metadata": {"generation": 2},
            "status": {
                "observedGeneration": 2,
                "conditions": [{"type": "Available", "status": "True"}],
            },
        }

    monkeypatch.setattr("app.components.service.get_resource", get)
    components = {
        key: ComponentRead(id=key, name=key, status="unverified", detail="")
        for key in ("envoy", "cloudnativepg", "workflows", "autoscaling")
    }
    service = ComponentsService(None, AuthService(Settings(env="test")))
    await service.kubernetes_probes(components, "platform-apps")
    assert components["envoy"].status == "unavailable"
    assert components["cloudnativepg"].status == "unverified"
    assert components["autoscaling"].status == "not_configured"
    assert components["workflows"].status == "ready"


async def test_trace_evidence_requires_matching_trace_with_spans(monkeypatch):
    trace_id = "0" + "a" * 31
    traces = [{"traceID": trace_id[1:], "spans": [{"operationName": "agent.run"}]}]
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": traces}))
    monkeypatch.setattr(
        "app.components.service.httpx.AsyncClient", lambda **_: original(transport=transport)
    )
    service = ComponentsService(None, AuthService(Settings(env="test")))
    components = {
        key: ComponentRead(id=key, name=key, status="unverified", detail="Health only")
        for key in ("jaeger", "otel")
    }
    await service.trace_evidence(components, trace_id)
    assert all("retrieved from Jaeger" in c.detail for c in components.values())
    traces.clear()
    for c in components.values():
        c.detail = "Health only"
    await service.trace_evidence(components, trace_id)
    assert all(c.detail == "Health only" for c in components.values())


async def test_encryption_evidence_requires_recent_completed_control_plane_check(monkeypatch):
    from datetime import UTC, datetime, timedelta

    data = {
        "provider": "k3s",
        "verified_at": datetime.now(UTC).isoformat(),
        "status": "Encryption Status: Enabled\nCurrent Rotation Stage: reencrypt_finished\n"
        "Server Encryption Hashes: All hashes match",
    }

    async def get(path):
        assert path.endswith("/configmaps/platform-encryption-evidence")
        return {"data": data}

    monkeypatch.setattr("app.components.service.get_resource", get)
    service = ComponentsService(None, AuthService(Settings(env="test")))
    component = ComponentRead(id="secrets", name="Secrets", status="unverified", detail="")
    await service.encryption_evidence(component, "platform-apps")
    assert component.status == "ready"
    for timestamp in (
        datetime.now(UTC) - timedelta(hours=25),
        datetime.now(UTC) + timedelta(hours=1),
    ):
        data["verified_at"] = timestamp.isoformat()
        await service.encryption_evidence(component, "platform-apps")
        assert component.status == "unverified"
    data["verified_at"] = datetime.now(UTC).isoformat()
    data["status"] = "Encryption Status: Enabled\nCurrent Rotation Stage: start"
    await service.encryption_evidence(component, "platform-apps")
    assert component.status == "unverified"

import asyncio

import httpx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import col, select

from app.components.schemas import ComponentRead
from app.kit.cache import Cache
from app.kit.db.postgres import transaction
from app.kit.kubernetes import condition, get_resource
from app.kit.objects import Objects
from app.models import Project, Run, Task

CATALOG = {
    "identity": "OIDC identity",
    "envoy": "Envoy Gateway",
    "valkey": "Valkey",
    "objects": "SeaweedFS",
    "workflows": "Argo Workflows",
    "otel": "OpenTelemetry",
    "prometheus": "Prometheus",
    "alertmanager": "Alertmanager",
    "jaeger": "Jaeger",
    "perses": "Perses",
    "opensearch": "OpenSearch",
    "autoscaling": "Autoscaling",
    "clickhouse": "ClickHouse warehouse",
    "cdc": "Postgres → ClickHouse CDC",
    "agents": "Pydantic AI workflows",
    "secrets": "Encrypted Kubernetes Secrets",
    "cloudnativepg": "CloudNativePG",
}
PROBES = {
    "prometheus": ("prometheus", 9090, "/-/ready"),
    "alertmanager": ("alertmanager", 9093, "/-/ready"),
    "jaeger": ("jaeger", 16686, "/"),
    "perses": ("perses", 8080, "/"),
    "opensearch": ("opensearch", 9200, "/_cluster/health"),
    "clickhouse": ("clickhouse", 8123, "/ping"),
    "otel": ("otel-collector", 13133, "/"),
}


class ComponentsService:
    def __init__(self, engine, auth):
        self.engine, self.auth = engine, auth

    async def list(self, principal, environment) -> list[ComponentRead]:
        self.auth.authorize(principal, "platform:read", environment)
        components = {
            key: ComponentRead(
                id=key,
                name=name,
                status="unverified",
                detail="No live evidence available in this environment",
            )
            for key, name in CATALOG.items()
        }
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            postgres = ComponentRead(
                id="postgres", name="PostgreSQL", status="ready", detail="SELECT 1 succeeded"
            )
        except (SQLAlchemyError, OSError, TimeoutError):
            postgres = ComponentRead(
                id="postgres",
                name="PostgreSQL",
                status="unavailable",
                detail="Database probe failed",
            )
        settings = self.auth.settings
        components["identity"].status = "unverified" if settings.oidc_jwks_url else "not_configured"
        components["identity"].detail = (
            "OIDC configured; interactive sign-in has not been verified"
            if settings.oidc_jwks_url
            else "Development tokens are active; no OIDC provider configured"
        )
        components["cdc"].status = "not_implemented"
        components[
            "cdc"
        ].detail = "Replication is not implemented; warehouse health does not prove CDC"
        components["secrets"].detail = (
            "Runtime Secrets are used; control-plane encryption at rest is not attested. "
            "This application cannot verify datastore encryption by reading Secret objects."
        )
        if settings.object_endpoint:
            await self.kubernetes_probes(components, settings.workflow_namespace)
            async with httpx.AsyncClient(timeout=3) as client:

                async def probe(key, service, port, path):
                    try:
                        response = await client.get(
                            f"http://{service}.platform-cluster.svc.cluster.local:{port}{path}"
                        )
                        response.raise_for_status()
                        components[key].status = "ready"
                        components[key].detail = (
                            f"Health endpoint returned HTTP {response.status_code}; "
                            "end-to-end delivery not verified"
                        )
                    except httpx.HTTPError:
                        components[key].status = "unavailable"
                        components[key].detail = "Service health request failed"

                await asyncio.gather(*(probe(key, *address) for key, address in PROBES.items()))
            try:
                async with Objects(settings).client() as client:
                    await client.head_bucket(Bucket=settings.object_bucket)
                components["objects"].status = "ready"
                components["objects"].detail = "Authenticated bucket access succeeded"
            except Exception:
                components["objects"].status = "unavailable"
                components["objects"].detail = "Bucket access failed"
            cache = Cache(settings)
            try:
                if cache.client and await cache.client.ping():
                    components["valkey"].status = "ready"
                    components[
                        "valkey"
                    ].detail = "Authenticated PING succeeded; run progress uses expiring keys"
            except Exception:
                components["valkey"].status = "unavailable"
                components["valkey"].detail = "Cache connection failed"
            finally:
                await cache.close()
            await self.agent_evidence(components, principal, environment)
            await self.metric_evidence(components)
        return [postgres, *components.values()]

    async def kubernetes_probes(self, components, namespace):
        probes = {
            "envoy": (
                "gateway.networking.k8s.io/v1",
                "platform-cluster",
                "gateways/platform",
                "Programmed",
            ),
            "cloudnativepg": (
                "postgresql.cnpg.io/v1",
                "platform-cluster",
                "clusters/platform-postgres",
                "Ready",
            ),
            "workflows": (
                "apps/v1",
                "platform-cluster",
                "deployments/workflow-controller",
                "Available",
            ),
            "autoscaling": (
                "autoscaling/v2",
                namespace,
                "horizontalpodautoscalers/platform-api",
                "ScalingActive",
            ),
        }

        async def probe(key, group, ns, resource, required):
            component = components[key]
            try:
                data = await get_resource(f"/apis/{group}/namespaces/{ns}/{resource}")
                ready = condition(data, required)
                if key == "workflows":
                    ready = ready and data.get("status", {}).get("observedGeneration", 0) >= data[
                        "metadata"
                    ].get("generation", 0)
                if key == "envoy":
                    ready = ready and condition(data, "Accepted")
                component.status = "ready" if ready else "unavailable"
                component.detail = f"Kubernetes {resource}: {required}={ready}"
                if key == "autoscaling" and ready:
                    component.detail += (
                        "; CPU metrics available; load-driven scaling not yet verified"
                    )
                if key == "workflows" and ready:
                    component.detail += (
                        "; controller ready, task completion evidence is reported under agents"
                    )
            except httpx.HTTPStatusError as exc:
                component.status = (
                    "not_configured" if exc.response.status_code == 404 else "unverified"
                )
                component.detail = (
                    f"Kubernetes status query returned HTTP {exc.response.status_code}"
                )
            except (httpx.HTTPError, OSError, ValueError):
                component.status = "unverified"
                component.detail = "Kubernetes status could not be read"

        await asyncio.gather(*(probe(key, *value) for key, value in probes.items()))

    async def agent_evidence(self, components, principal, environment):
        try:
            async with transaction(self.engine) as session:
                run = (
                    await session.exec(
                        select(Run)
                        .join(Task, col(Task.id) == col(Run.task_id))
                        .join(Project, col(Project.id) == col(Task.project_id))
                        .where(
                            Project.owner == principal.subject,
                            Project.environment == environment,
                            col(Project.deleted_at).is_(None),
                            col(Task.deleted_at).is_(None),
                            col(Run.deleted_at).is_(None),
                            Run.status == "completed",
                            col(Run.workflow_name).is_not(None),
                            col(Run.trace_id).is_not(None),
                        )
                        .order_by(col(Run.created_at).desc())
                        .limit(1)
                    )
                ).first()
            if run:
                components["agents"].status = "ready"
                components["agents"].detail = (
                    f"Completed workflow {run.workflow_name} at {run.updated_at}; "
                    "historical execution evidence, not model-provider or current worker health"
                )
                await self.trace_evidence(components, run.trace_id)
            else:
                components[
                    "agents"
                ].detail = (
                    "No completed workflow evidence for your projects yet; run an inspector task"
                )
        except SQLAlchemyError:
            components["agents"].status = "unavailable"
            components["agents"].detail = "Could not read execution evidence"

    async def trace_evidence(self, components, trace_id):
        if not trace_id or len(trace_id) != 32:
            return
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(
                    f"http://jaeger.platform-cluster.svc.cluster.local:16686/api/traces/{trace_id}"
                )
                response.raise_for_status()
                traces = response.json().get("data", [])
                found = any(
                    t.get("traceID", "").lstrip("0") == trace_id.lstrip("0") and t.get("spans")
                    for t in traces
                )
            if found:
                for key in ("jaeger", "otel"):
                    components[key].status = "ready"
                    components[key].detail = f"Agent trace {trace_id} retrieved from Jaeger"
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return  # Preserve health evidence; failed retrieval is not proof of delivery.

    async def metric_evidence(self, components):
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(
                    "http://prometheus.platform-cluster.svc.cluster.local:9090/api/v1/query",
                    params={"query": 'sum(platform_http_requests_total{job="platform-api"})'},
                )
                response.raise_for_status()
                result = response.json().get("data", {}).get("result", [])
            if result and float(result[0]["value"][1]) > 0:
                components[
                    "prometheus"
                ].detail = "Prometheus query returned scraped API request metrics"
                components["prometheus"].status = "ready"
            elif components["prometheus"].status == "ready":
                components["prometheus"].status = "unverified"
                components[
                    "prometheus"
                ].detail = "Prometheus is healthy but no scraped API request metrics were found"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            components["prometheus"].status = "unverified"
            components["prometheus"].detail = "Could not verify API metric ingestion"

# Kubernetes platform starter pack

An experimentation workspace for giving agents tasks, following up with instructions, and inspecting the platform that executes them. The application runs on portable Kubernetes resources; an existing Colima Kubernetes cluster runs the full local stack. No kind.

**Current demo:** projects, tasks, conversations, reports, cancellation, and soft deletion are implemented. A live Colima walkthrough has completed a task and a follow-up through Argo, persisted metadata in Postgres and conversation/report objects in SeaweedFS, observed Valkey progress, and recorded trace IDs. The default inspector uses Pydantic AI's deterministic TestModel with real platform probes, clearly labeled in the UI. It does not call a language model.

## Run locally

Use the existing default Colima profile with Kubernetes enabled and the Docker runtime. Install Docker, kubectl, Bun, and uv. Local commands currently use Bun and uv as well as Docker and kubectl; a Docker-and-kubectl-only bootstrap is not yet complete. The checked-in tool versions and frozen lockfiles live under server/ and web/.

```sh
./bin/install
./bin/local-check
./bin/local-up
./bin/local-forward
```

Keep forwarding running. In another terminal:

```sh
./bin/local-credentials
```

Open https://web.localhost:5173, choose Sign in, and use the generated developer or viewer credentials. Keycloak is available at https://auth.localhost:5173. API documentation is at https://api.localhost:8000/docs and MCP at https://mcp.localhost:8001/mcp. Stop native development servers occupying those ports first. If the local CA is not trusted yet, run `bin/certs-trust` once. No hosts-file changes are needed.

Create a project, start a task with **Platform inspector**, read its report, and send a follow-up. The run details show Argo workflow names, trace IDs, token usage, and short-lived progress. The System view probes the backing services and distinguishes implemented checks from unverified integrations. Appearance supports System, Light, and Dark.

`bin/local-experiment` performs the task/follow-up walkthrough against the running HTTPS API and leaves a project to inspect in the UI. `bin/local-status` shows workloads. `bin/local-stop` retains volumes and credentials; `bin/local-up` resumes them. See the [local profile](k8s/profiles/local/README.md) for capacity and operational limits.

## Repository layout

```text
web/                         React, Vite, shadcn/ui, Hey API SDK
server/
  app/
    api.py                   FastAPI entrypoint
    mcp.py                   FastMCP entrypoint
    worker.py                Dispatcher and Argo agent entrypoint
    auth/                    Shared authentication and authorization
    projects/                router.py, service.py, schemas.py
    exports/                 Snapshot requests, lifecycle, and execution
    tasks/                   router.py, service.py, schemas.py
    agents/                  Pydantic AI execution
    models/                  SQLModel metadata tables
    kit/                     Database, objects, cache, workflows, telemetry
  migrations/                Alembic: timestamp_ms_rev_slug.py
bin/                         Install, develop, format, test, build, deploy
k8s/
  base/
    apps/                    Web, API/MCP, dispatcher, worker, migrations
    cluster/                 Platform services and controller configuration
    vendor/                  Pinned upstream manifests and image inventory
  profiles/
    local/                   Existing Colima Kubernetes
    test/
    preview/
    production/
```

Each section has its own README. Start with [commands](bin/README.md), [server](server/README.md), [web](web/README.md), [models](server/app/models/README.md), [migrations](server/migrations/README.md), and [Kubernetes](k8s/README.md).

## Storage and execution

| System | Responsibility |
| --- | --- |
| Postgres / CloudNativePG | Project/task ownership, durable run queue and status, object references, lifecycle timestamps, usage, and events. SQLModel + asyncpg; Alembic owns schema changes. |
| SeaweedFS | Immutable user messages, agent replies, Pydantic AI conversation history, and reports. SQL stores keys, byte counts, and checksums. The async S3 client targets SeaweedFS explicitly; no AWS service or credentials are required. |
| Valkey | Reconstructible run progress and notifications with TTLs. Cache failure does not lose queued work or conversations. |
| Argo Workflows | Executes one bounded agent run per workflow. The dispatcher claims Postgres work, serializes runs per task, and reconciles workflow state. |
| Pydantic AI | Agent/tool execution inside the server image. No separate agent infrastructure vendor. |

Models use prefixed ULIDs and created_at, updated_at, deleted_at. Services enforce project ownership and environment boundaries for API and MCP alike. Soft deletion hides records and cancels outstanding work. Uploads precede SQL references, so failed database transactions can leave unreferenced objects; retention and orphan cleanup remain to be implemented.

The inspector works without external credentials. Research and analysis agents become selectable when `APP_AGENT_MODEL` is configured. Provider credentials must also be supplied to the worker through Kubernetes Secrets. Both currently use bounded platform inspection tools; arbitrary code execution and general developer-agent capabilities are not implemented. Provider-backed execution has not yet been validated.

## Platform services

The local composition includes Envoy Gateway and Gateway API, cert-manager, CloudNativePG, Valkey, SeaweedFS, Argo Workflows, Prometheus, Alertmanager, Jaeger, Perses, OpenSearch, OpenTelemetry Collector, and ClickHouse. Application workloads use `platform-apps`; platform services and operators use `platform-cluster`. Kubernetes system namespaces remain separate.

Application traces use OpenTelemetry with message contents excluded. The API exposes Prometheus request metrics. The catalog queries Prometheus for scraped API metrics and attempts to retrieve a completed agent trace from Jaeger. Other service probes remain health checks; they do not prove the complete monitoring pipeline. The local API HPA uses existing CPU metrics, and the catalog reads its ScalingActive condition. Load-driven scaling, log ingestion, curated dashboards/alerts, still need end-to-end validation. Warehouse freshness is reported from the latest validated batch export.

The base uses ordinary Kubernetes resources for telemetry, dashboards, logs, and warehousing, avoiding extra operators. Kueue, KEDA, PeerDB, and additional service operators remain options for later consideration. Reuse an existing Metrics Server; do not install a duplicate.

Kubernetes Secrets supply runtime credentials. Encryption at rest is a cluster prerequisite, not a property of Secret YAML. The local bootstrap does not configure Colima's control-plane encryption. Local single-node data services and internal OpenSearch without its security plugin are development settings, not production defaults to adopt without review.

## Development and validation

All commands live in `bin/`. Installs use frozen lockfiles; dependency changes require exact pins and validation. `bin/web-ui add COMPONENT` runs `bunx --bun shadcn@4.21.0`, then pins any introduced dependencies and installs the resulting lockfile frozen. React Router loaders/actions/fetchers handle requests and revalidation; there is no TanStack dependency.

```sh
./bin/check
./bin/client-check
./bin/postgres-start
./bin/integration-test
./bin/local-check
./bin/local-experiment
```

The separate Docker Postgres is for native development and disposable integration databases; it does not require Kubernetes and is independent of local CloudNativePG. Native `server-dev`, `mcp-dev`, and `web-dev` commands remain available, but the full task demo requires the platform services and dispatcher.

Integration tests cover migrations, persistence, API/MCP authentication, owner isolation, concurrent run claims, cancellation, conversations, and object-storage failure behavior. The local walkthrough adds real SeaweedFS, Valkey, and Argo execution. Remote deployment, real model calls, backup/restore, and the remaining telemetry integrations are not yet validated.

## Vendored components and delivery

The [vendor inventory](k8s/base/vendor/README.md) records pinned inputs, checksums, licenses, and upgrade procedures. Keep upstream manifests unchanged and place configuration in cluster/apps bases and profiles. The [shadcn source](web/src/components/ui/README.md) records its generator and provenance. Review generated source and lockfile diffs together; a pinned CLI does not make its remote registry immutable.

GitHub Actions delivery, preview lifecycle, and production promotion remain to be implemented; this checkout has no starter-pack workflow. Cluster provisioning is independent of this example. Supply an existing Kubernetes context.

Before remote deployment, configure OIDC, immutable image references, storage and backup destinations, certificate/DNS integration, resource sizing, and deployment identity. Prove the full local workflow first, then repeat it on the target cluster and validate the remaining telemetry, recovery, and scaling paths.

## Identity and warehouse demo

Local sign-in uses Keycloak authorization code with PKCE. Developer accounts can create projects/tasks and exports; viewers have read-only access. bin/local-token [--viewer] [--mcp] obtains a short-lived Keycloak access token for CLI checks. It requires local-forward and reads only the generated demo credentials from the cluster. Native development outside Kubernetes still supports bin/dev-token.

In Warehouse, choose **Export now**. The same metadata snapshot worker runs nightly at **02:00 UTC** through Argo. Export history shows state, workflow, manifest key, freshness, and ClickHouse row counts. bin/local-export checks submission idempotency, manifest/count agreement, and viewer denial. bin/local-nightly invokes the scheduled entrypoint immediately; bin/local-identity-check verifies API/MCP audiences and shows completed nightly export evidence. See [exports](server/app/exports/README.md) and [Keycloak](k8s/base/cluster/keycloak/README.md).

## Cleanup

`bin/local-cleanup` previews a data-preserving stop; `bin/local-cleanup --apply` stops workloads and suspends nightly exports while retaining data. Resume with `bin/local-up`.

`bin/local-cleanup --delete-data` previews a full project uninstall. Add `--apply` to delete both project namespaces and their Secrets/PVCs, all project custom resources, controllers, CRDs, cluster roles/bindings, admission webhooks/policies, routing classes, and priority classes. It waits for custom-resource finalizers while controllers are alive, then removes controllers and cluster-wide definitions. Projects, conversations, warehouse snapshots, Keycloak users, and service credentials are removed.

The preflight refuses to remove CRDs used outside the project, cluster roles referenced by external bindings, global resources without project ownership evidence, or volumes whose reclaim policy retains backing storage. Resolve those dependencies before uninstalling. The cluster and its own storage provisioner remain. Cluster-wide Secret encryption stays enabled; cleanup does not remove Colima, VM configuration, local files/certificates, Docker images, or the separate Docker development database. No live uninstall is performed by a preview.

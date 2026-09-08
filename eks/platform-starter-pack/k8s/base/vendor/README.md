# Vendor inventory

| Component | Release | Purpose |
| --- | --- | --- |
| [CloudNativePG](cloudnative-pg/README.md) | 1.30.0 | Postgres operator |
| [cert-manager](cert-manager/README.md) | 1.21.1 | Certificates |
| [Envoy Gateway](envoy-gateway/README.md) | 1.9.1 | Gateway controller and bundled Gateway API definitions |
| [Argo Workflows](argo-workflows/README.md) | 4.1.2 | Long-running workflows and agent tasks |
| [Gateway API](gateway-api/README.md) | 1.6.2 | Standalone standard definitions retained for upgrade review |

These vendors plus Argo Workflows are downloaded so far; the full initial deployment inventory is being prepared. Envoy already bundles Gateway API resources: its first composition should own those definitions, rather than installing both overlapping bundles. Vendor copies render independently; that does not prove combined compatibility or deployment readiness.

Run bin/vendor-check for SHA-256 verification, bin/vendor-fetch [component] to restore pinned files, and bin/krm-check to render the current inputs and compositions. sources.txt in each folder records URLs and checksums. Keep upstream files unchanged and put our overrides in ../cluster/ or ../apps/. Review release/license changes and CRD/RBAC/image diffs before upgrades. Signature verification and remaining container image digest overrides are pending.

Required in the first deployment: Valkey, SeaweedFS, Argo Workflows, Prometheus/Alertmanager, Jaeger, Perses, OpenSearch, OpenTelemetry Collector, ClickHouse/CDC, and Pydantic AI tasks on Argo. No Agent Substrate or Agent Sandbox vendor is included. Identity remains an explicit decision. For further consideration only: Kueue, KEDA, duplicate Metrics Server, additional service operators, Jaeger chart, or PeerDB are outside the initial set. Pydantic AI and language libraries remain in the application lockfiles.

The local Docker Postgres runtime image remains pinned in bin/postgres-start. Hey API-generated client support is regenerated from the pinned web lockfile through bin/generate-client. Neither requires copying another Kubernetes controller here.

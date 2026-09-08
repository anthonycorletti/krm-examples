# Components

The catalog distinguishes ready, unavailable, unverified, not_configured, and not_implemented. Kubernetes status probes check the configured Gateway, CNPG cluster, Argo controller, and API HPA with read-only, named-resource RBAC. Agent execution evidence is restricted to the caller's own non-deleted projects and explicitly identifies the completed workflow and time.

A ready HTTP endpoint is only a health check; it does not prove delivery of traces, logs, or alerts. OIDC authentication and validated warehouse snapshots provide integration evidence; unverified control-plane Secret encryption is reported explicitly. The app does not read Kubernetes Secret objects to attempt to infer encryption at rest.

GET /api/components and MCP list_components share this service and authorization. Test with bin/server-check and bin/integration-test. Deploy cluster/inspection alongside apps in each profile that uses Kubernetes probes.

Secret encryption is reported from an administrator-written platform-encryption-evidence ConfigMap. Only completed K3s reencryption with matching server hashes and a timestamp less than 24 hours old is ready. Missing, stale, or invalid evidence is unverified. The app has get access only to this named ConfigMap, never encryption keys or Secret values.

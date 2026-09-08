# Components

The catalog distinguishes ready, unavailable, unverified, not_configured, and not_implemented. Kubernetes status probes check the configured Gateway, CNPG cluster, Argo controller, and API HPA with read-only, named-resource RBAC. Agent execution evidence is restricted to the caller's own non-deleted projects and explicitly identifies the completed workflow and time.

A ready HTTP endpoint is only a health check; it does not prove delivery of traces, logs, or alerts. OIDC configuration, missing CDC, and unverified control-plane Secret encryption are reported explicitly. The app does not read Kubernetes Secret objects to attempt to infer encryption at rest.

GET /api/components and MCP list_components share this service and authorization. Test with bin/server-check and bin/integration-test. Deploy cluster/inspection alongside apps in each profile that uses Kubernetes probes.

# Application workloads

Only developer-owned application resources belong here: web, API/MCP, workers, and migration Jobs. These run in platform-apps. Databases, platform services, and their operators belong in ../cluster/ and run in platform-cluster.

The root composition contains web/API/MCP deployments, services, configuration, and routes. migrations/ contains the Alembic Job; workflows/ contains the Pydantic AI inspection WorkflowTemplate and execution RBAC. Apply these separate stages only after their controllers and configuration are ready. Full EKS deployment is unverified.

Cross-namespace connections use explicit service DNS names and scoped network policies. Kubernetes Secret references are namespace-local, so application credentials must be provisioned into platform-apps; Pods cannot directly mount a Secret from platform-cluster.

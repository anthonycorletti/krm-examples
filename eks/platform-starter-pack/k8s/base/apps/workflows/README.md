# Agent workflows

platform-inspect runs app.worker from the server image with Pydantic AI TestModel and a read-only application principal. Argo handles task lifecycle; no agent-specific infrastructure is installed.

Prerequisites: Argo controller/CRDs, platform-settings, platform-postgres-app and platform-postgres-ca in platform-apps, reachable Postgres, and a published server image. Apply this directory after controller readiness. Run `bin/workflow-demo CONTEXT IMAGE@sha256:DIGEST` to submit, wait, and print pod logs. This command does not install prerequisites.

Logs remain until the workflow expires after one day. Infrastructure retries are limited to one; the task has a 60-second timeout and the workflow a five-minute deadline. Success means the task ran; inspect JSON for component health. EKS verification remains pending.

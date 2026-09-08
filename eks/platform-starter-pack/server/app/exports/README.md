# Metadata exports

POST /api/exports queues a full snapshot of the caller's project metadata. An idempotency key is required and is scoped by owner and environment. Reads and manifests enforce the same ownership. API and MCP use ExportsService; viewers cannot start exports.

Argo runs export workers from the server image. Postgres stores lifecycle/status and counts in the Export SQLModel table (migration 0003). The worker takes a repeatable-read snapshot of Project, Task, Run, Message, Event, and Artifact metadata, including soft-deleted records. Chat bodies, model history, credentials, and the Keycloak database are not exported.

The worker writes zstd-compressed Parquet and a manifest with SHA-256, size, owner, environment, and row counts to SeaweedFS. Once the manifest exists, retries use those exact objects, validate the checksum, rebuild an unpublished ClickHouse table for that export, and compare counts. Only then is the SQL export marked completed. The warehouse API queries the newest completed export for the caller; failures retain the previous successful snapshot. Snapshot tables are named from validated prefixed ULIDs, never user-provided SQL identifiers.

The nightly CronWorkflow runs at 02:00 UTC. Its scheduler queues one export for each owner in the configured environment; retries use the workflow name as a stable request key. Export now uses the same worker. The local profile requires no CDC service. CDC remains an option if future analytics need lower latency.

This demo bounds each export to 100,000 rows and 64 MiB of Parquet. Export objects and snapshot tables are retained; automated retention/orphan cleanup is not yet implemented. It is a metadata analytics snapshot, not a Postgres backup. Use bin/local-export for live validation and bin/integration-test for ownership, soft deletion, idempotency, and retry tests.

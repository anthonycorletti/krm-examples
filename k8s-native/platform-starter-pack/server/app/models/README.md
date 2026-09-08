# SQLModel tables

Project, Task, Run, Message, Event, and Artifact describe experimentation metadata. Export records snapshot ownership, lifecycle, manifest location, and validated row counts; migration 0003 adds this table. New records use ULID string IDs, prefixed for these topics, plus created_at, updated_at, and deleted_at. Project ownership and filtering of soft-deleted records are enforced by the application services.

Postgres owns metadata and durable execution state. Object storage owns conversation bodies, Pydantic AI history, attachments, and tool output; SQL holds keys, sizes, and checksums. Upload an immutable object before committing its SQL reference, with unreferenced-object cleanup after a retention window still to implement. Valkey holds reconstructible cache entries, short-lived progress and notification fan-out; rate limiting is not implemented; it is never the only record of queued work or conversations.

Migration 0002 preserves existing verification records while changing their ID storage type to text and adding lifecycle fields. Historical identifiers remain unchanged. Downgrade removes experimentation tables and maps new verification ULIDs to deterministic UUIDs for compatibility with the old schema.

Verification stores the component, environment, initiator, time, and evidence of a committed Postgres check. __init__.py is the table registry imported by Alembic. Public schemas live beside their topics and explicitly choose fields to expose.

After changing a table, run bin/migration-new "description", review the generated revision, then bin/migrate and bin/migration-check against the development database. Run bin/integration-test before delivery.

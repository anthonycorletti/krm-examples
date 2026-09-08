# Migration revisions

Committed Alembic revisions use {timestamp_ms}_{rev}_{slug}.py. Create them with bin/migration-new, review the operations, and retain stable revision IDs once released. 0001 introduces platform verification evidence.

Run bin/integration-test and bin/migration-check after a change.

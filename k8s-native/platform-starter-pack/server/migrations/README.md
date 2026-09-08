# Alembic migrations

Use bin/migration-new "description" to generate a candidate revision named {timestamp_ms}_{rev}_{slug}.py. The prefix is the actual Unix timestamp in milliseconds at generation time. Revision IDs and down_revision links control ordering; filenames aid navigation.

bin/migrate applies head. bin/migration-check compares SQLModel metadata with the database. env.py imports all table models and uses an asyncpg engine, with Alembic's run_sync bridge and a transaction-scoped advisory lock. Revision functions remain synchronous because Alembic's operation API is synchronous; database I/O uses the async engine bridge.

Review autogeneration before applying: renames, data backfills, rollout compatibility, and irreversible changes need deliberate handling. Use bin/integration-test for fresh installs, repeated upgrades, drift, persistence, and disposable downgrade/re-upgrade checks. Production downgrade is not automated.

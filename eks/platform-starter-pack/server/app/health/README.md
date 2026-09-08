# Health

/livez confirms the process can respond. /readyz performs an async query of alembic_version and checks it against the release head; it returns 503 if the database is unavailable or requires migration. These probes do not enumerate the wider platform and expose no credentials.

Run bin/integration-test to verify readiness before and after migrations.

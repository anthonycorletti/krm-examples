# Verification

POST /api/verifications and the MCP run_verification tool write a SQLModel record, commit, and read it through a new async session. The operation requires platform:verify for the selected environment. History and migration status require platform:read.

GET /api/verifications returns at most the latest 50 records. GET /api/migrations compares the applied Alembic revision with the image's expected head. Successful records are durable; failed attempts are surfaced as errors and are not yet stored as separate run records.

bin/integration-test checks committed visibility, restart persistence, migration upgrades/downgrades in disposable databases, and HTTP/MCP permissions. A console-triggered isolated migration workflow remains planned.

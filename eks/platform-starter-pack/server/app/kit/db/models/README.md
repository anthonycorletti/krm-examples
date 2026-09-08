# Shared model bases

RecordModel provides a UUID primary key and timezone-aware creation timestamp. Concrete table models live in app/models/. Add reusable persistence fields here without importing application topics. Alembic owns schema creation; validate changes with bin/migration-check and bin/integration-test.

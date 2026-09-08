# Database infrastructure

postgres.py builds an AsyncEngine using postgresql+asyncpg URLs and SQLModel AsyncSession factories. transaction() opens a session and commits or rolls back through an async context manager. Each operation owns its session; never share a mutable session between concurrent tasks. Engine cleanup is awaited at process shutdown. Connection and statement timeouts bound probes.

Run bin/postgres-start, bin/migrate, and bin/integration-test. No runtime create_all() path exists.

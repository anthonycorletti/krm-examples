# Tests

bin/server-test runs unit tests; database integration tests skip unless APP_TEST_DATABASE_URL is explicitly set. bin/integration-test starts the owned development Postgres container and enables those tests.

Each integration test creates a uniquely named database and drops only that database on teardown. Coverage includes missing/expired/wrong-audience tokens, scope/environment denials, migration bootstrap and model drift, evidence persistence across application lifespans, real HTTP MCP, and an async Pydantic AI tool using TestModel.

The development database role can create/drop databases for test isolation. Do not point this suite at production.

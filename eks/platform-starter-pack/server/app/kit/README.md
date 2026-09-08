# Kit

Shared infrastructure lives here: async database lifecycle in db/, SQLModel record bases in db/models/, immutable object I/O in objects.py, disposable Valkey progress in cache.py, Kubernetes workflow requests in workflows.py, OpenTelemetry setup in telemetry.py, and Prometheus instruments in metrics.py. Utilities and transport-neutral errors stay in utils.py and errors.py.

Kit does not import topic routers or services. Configuration lives in app/settings.py. Clients use explicit endpoints and credentials; S3 requests target the configured object service rather than AWS discovery. Verify through bin/server-typecheck, bin/integration-test, and the real-service bin/local-experiment walkthrough.

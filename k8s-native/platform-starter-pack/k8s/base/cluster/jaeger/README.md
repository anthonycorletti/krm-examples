# jaeger

Receives traces through the OpenTelemetry Collector and stores them in OpenSearch. The application excludes conversation content from agent instrumentation. Run details expose trace IDs; verify trace retrieval separately from service readiness.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the local K3s composition.

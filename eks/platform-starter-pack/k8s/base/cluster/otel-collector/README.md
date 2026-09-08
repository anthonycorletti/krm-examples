# otel-collector

Receives OTLP traces, metrics, and logs; exports traces to Jaeger, metrics to its Prometheus endpoint, and logs to OpenSearch. Applications currently emit traces and expose HTTP metrics directly. Application log export is not yet wired.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

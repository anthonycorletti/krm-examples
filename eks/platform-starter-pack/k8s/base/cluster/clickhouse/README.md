# clickhouse

Single-instance warehouse service. Readiness is probed by the System view. Postgres CDC has not yet been implemented or validated; a healthy ClickHouse pod is not evidence of replication.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

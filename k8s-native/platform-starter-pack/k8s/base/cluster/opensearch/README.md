# opensearch

Stores Jaeger traces and is the intended application log destination. The local single-node service disables the security plugin and is not exposed through the public gateway. Configure authentication, TLS, retention, and recovery for remote environments.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

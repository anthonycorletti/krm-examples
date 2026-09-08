# cert-manager

Composes the pinned vendored cert-manager controller into platform-cluster. Issuers and Gateway certificates live in the adjacent gateway section; local development uses its existing trusted CA.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

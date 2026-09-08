# CloudNativePG

Vendored release: [1.30.0](https://github.com/cloudnative-pg/cloudnative-pg/releases/tag/v1.30.0), commit `4b5e244a7d031f67e025c83c1555e7726ecbbfa1`. License: Apache-2.0, copied in LICENSE. upstream.yaml is the unmodified release asset. sources.txt records exact download URLs and SHA-256 checksums; the manifest checksum matches the GitHub release asset metadata.

Run bin/vendor-check to verify copies offline or bin/vendor-fetch to restore the pinned copies. Fetching never selects a newer version automatically. Checksums verify the pinned bytes; signature verification has not been implemented.

Operator image: `ghcr.io/cloudnative-pg/cloudnative-pg:1.30.0@sha256:a2701eb97cdd2a34b1fdb2cb51987f544b706e40bec72ae7146cd8580efefebb`.

Database image: `ghcr.io/cloudnative-pg/postgresql:17.9-system-trixie@sha256:8fd8db26090e602458269741849e5c9cd7aff577a22af53c8a402c580fd1e729`.

The upstream manifest owns postgresql.cnpg.io CRDs, cluster RBAC, admission webhooks, and the cnpg-system namespace/controller. Our image overrides live in ../../cluster/cloudnative-pg/. The [upstream compatibility table](https://cloudnative-pg.io/docs/1.30/supported_releases/) lists Kubernetes 1.34–1.36 for this series. EKS compatibility and runtime behavior must be checked against the selected target before claiming support.

To upgrade, review release notes and compatibility, update sources.txt with reviewed release URLs and checksums, run bin/vendor-fetch, update image digests in the base, and run bin/vendor-check and bin/krm-check. Review the upstream CRD/RBAC diff. Then test installation, migrations, persistence, and recovery on EKS before promotion. Do not uninstall CRDs as an upgrade step: deleting them can delete managed resources.

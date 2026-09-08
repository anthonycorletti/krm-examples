# Postgres

CloudNativePG Cluster resources for the EKS deployment test. The shared base requests three instances and 10 GiB per instance. The test profile must select its storage class and sizing after the EKS target is identified. These resources have been rendered, not deployed or verified on EKS.

The database image is PostgreSQL 17.9 on Trixie, pinned by digest. CloudNativePG creates application credentials in a Kubernetes Secret; superuser access is disabled. Credentials are not committed to Git. Storage encryption, backups, and recovery verification still need the EKS profile configuration.

The operator is a separate installation stage in ../cloudnative-pg/. Local bin/postgres-start continues to use Docker and does not depend on these manifests.

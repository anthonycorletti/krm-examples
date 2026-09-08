# Postgres

CloudNativePG Cluster resources for Kubernetes deployments. The shared base requests three instances and 10 GiB per instance. The test profile must select its storage class and sizing after target storage is identified. The local profile is validated on Colima; remote deployment remains unverified.

The database image is PostgreSQL 17.9 on Trixie, pinned by digest. CloudNativePG creates application credentials in a Kubernetes Secret; superuser access is disabled. Credentials are not committed to Git. Storage encryption, backups, and recovery verification still need target-specific profile configuration.

The operator is a separate installation stage in ../cloudnative-pg/. Local bin/postgres-start continues to use Docker and does not depend on these manifests.

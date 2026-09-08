# Base

- apps/: developer-owned workloads in platform-apps.
- cluster/: Postgres and other platform services, their operators, networking, certificates, namespaces, and scaling configuration. Namespaced services run in platform-cluster.
- vendor/: unmodified, pinned upstream files with licenses and retrieval checksums.

Profiles in ../profiles/ supply environment differences. The first deployment target is EKS. Raw vendor bundles and Postgres compositions render locally; live deployment validation remains pending.

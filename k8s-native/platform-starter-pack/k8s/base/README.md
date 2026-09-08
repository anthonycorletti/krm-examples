# Base

- apps/: developer-owned workloads in platform-apps.
- cluster/: Postgres and other platform services, their operators, networking, certificates, namespaces, and scaling configuration. Namespaced services run in platform-cluster.
- vendor/: unmodified, pinned upstream files with licenses and retrieval checksums.

Profiles in ../profiles/ supply environment differences. The local K3s profile is implemented. Vendor bundles and service compositions render locally and have been exercised on the local K3s cluster; remote deployment validation remains pending.

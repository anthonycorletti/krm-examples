# Base

- apps/: developer-owned workloads in platform-apps.
- cluster/: Postgres and other platform services, their operators, networking, certificates, namespaces, and scaling configuration. Namespaced services run in platform-cluster.
- vendor/: unmodified, pinned upstream files with licenses and retrieval checksums.

Profiles in ../profiles/ supply environment differences. The local Colima profile is implemented. Vendor bundles and service compositions render locally and have been exercised on Colima; remote deployment validation remains pending.

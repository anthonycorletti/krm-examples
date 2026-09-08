# Postgres operator

Shared CloudNativePG installation composed from the unmodified vendor manifest. This layer pins both the controller image and the instance-manager image reference by digest.

It installs cluster-wide CRDs, RBAC, and admission webhooks, plus the controller in platform-cluster. Install and wait for the operator before applying database resources. Review existing CNPG ownership before applying to a shared EKS cluster; operator upgrades affect every database it manages.

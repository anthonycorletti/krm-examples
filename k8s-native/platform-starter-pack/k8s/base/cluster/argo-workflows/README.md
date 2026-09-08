# Argo Workflows

Controller and internal server run in platform-cluster and manage workflows in platform-apps. The server uses Kubernetes client authentication and has no public route. Application OIDC tokens are not Argo credentials. Upstream cluster RBAC remains; task pods use a separate minimal service account.

Apply namespaces and this composition using server-side apply (large CRDs), wait for CRDs/controller readiness, then install apps/workflows. Local rendering is checked; Remote execution is pending.

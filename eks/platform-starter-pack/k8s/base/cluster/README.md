# Platform services

Cluster services and their operators belong here: Postgres, ingress, certificates, namespaces, and later caching, storage, workflows, observability, and scaling configuration. Namespaced platform resources run in platform-cluster. Developer-owned workloads run in platform-apps. namespaces.yaml declares both namespaces.

cloudnative-pg/ composes the operator from the unmodified vendor bundle, replacing its default namespace with platform-cluster. postgres/ contains its database Cluster resource in the same namespace. Apply namespaces first, then controllers/CRDs and readiness checks, then custom resources. There is intentionally no single apply for all stages.

The first deployment includes SQL, KV caching, object storage, workflows, monitoring, warehousing/CDC, and Argo execution for Pydantic AI tasks, alongside ingress and certificates. Use ordinary Kubernetes resources for services that do not need their own operator. Kueue, KEDA, separate monitoring/warehouse operators, and PeerDB remain for further consideration. Upstream files remain in ../vendor/; namespace and configuration changes belong here. Upstream default namespaces are not the platform namespace design.

The two-namespace layout applies to platform-owned workloads. Kubernetes/EKS system namespaces remain managed by the cluster. Isolated preview/test copies will need a separately documented namespace strategy before concurrent deployments are supported.

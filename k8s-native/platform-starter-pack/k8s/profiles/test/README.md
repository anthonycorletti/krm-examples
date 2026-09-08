# Test profile

Planned: run deployment tests against an explicitly selected Kubernetes context. The local K3s cluster is currently validated; remote deployment profiles still need implementation.

Choose the Kubernetes version, compute architecture, registry access, storage class, ingress addresses, TLS issuer, and encryption evidence for the target cluster. Preserve platform-apps and platform-cluster separation. Concurrent tests require an isolation strategy before sharing a cluster.

Cleanup must remove only resources owned by the test run, with an explicit data-retention choice. Shared operators, CRDs, and cluster provisioning remain outside workload cleanup.

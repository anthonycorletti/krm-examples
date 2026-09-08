# envoy-gateway

Pinned release: [envoyproxy/gateway v1.9.1](https://github.com/envoyproxy/gateway/releases/tag/v1.9.1). upstream.yaml is the unmodified release asset; its SHA-256 matches GitHub release metadata. LICENSE is the Apache-2.0 notice copied from the release tag. Exact URLs and checksums are in sources.txt. Signature verification and a complete container-image license inventory remain pending.

Resource ownership: Envoy Gateway controller, namespace/RBAC, and bundled Envoy/Gateway API CRDs.

Run bin/vendor-check to verify copies, or bin/vendor-fetch envoy-gateway to restore these exact versions. To upgrade, review the upstream release and compatibility requirements, update release URLs/checksums and this README, fetch the pinned files, review the CRD/RBAC/image diff, and run bin/krm-check. Keep deployment overrides under cluster/; do not edit upstream.yaml. Validate the composed configuration on the target cluster before promotion.

This bundle renders independently; installation, readiness, image digest overrides, and integration verification remain pending. Envoy's install bundle already includes Gateway API resources: do not apply the separate gateway-api bundle alongside it. The first ingress composition should use Envoy's bundled definitions as the single owner; the standalone Gateway API copy is a reference for explicit future upgrades.

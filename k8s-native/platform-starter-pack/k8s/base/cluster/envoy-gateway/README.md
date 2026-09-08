# envoy-gateway

Composes the pinned Envoy Gateway and Gateway API resources. The adjacent gateway section owns GatewayClass and routing/certificate configuration. Local routing uses a ClusterIP gateway through bin/local-forward.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

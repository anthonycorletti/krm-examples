# gateway

GatewayClass, Gateway, certificate, and issuer configuration. Profiles supply environment-specific hostnames, certificates, and exposure. The local overlay routes web.localhost, api.localhost, and mcp.localhost through Envoy.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the Colima composition.

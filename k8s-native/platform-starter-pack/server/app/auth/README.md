# Auth

One topic owns authentication (identity) and authorization (permissions). API and MCP validate their respective token audiences and reuse the same service policy checks. platform:read permits inspection and owned project/task reads; platform:verify permits project/task mutations and bounded verification. Environment and owner checks run before access.

Colima uses Keycloak OIDC tokens from browser PKCE or bin/local-token. Native local/test development supports short-lived symmetric tokens from bin/dev-token. Remote environments require configured OIDC issuer/JWKS validation; browser PKCE and MCP discovery support are present, with remote-provider validation still pending. Verify with bin/server-check and bin/integration-test.

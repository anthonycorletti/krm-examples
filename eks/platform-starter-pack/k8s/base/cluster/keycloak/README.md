# Keycloak

One pinned Keycloak 26.7.3 deployment uses a separate keycloak database and role in CloudNativePG. Kubernetes Secrets provide database, bootstrap admin, and local developer/viewer credentials. No operator or AWS service is required.

The checked-in realm defines platform-web (authorization code + PKCE), local platform-cli and platform-mcp-cli clients, API/MCP audiences, and the environment claim. Client-scope role mappings restrict platform:verify to developers. The local CLI password grant is a test/development convenience; disable these clients for remote profiles. Use bin/local-credentials to read the generated local login passwords. Existing local-developer subject IDs are retained so earlier projects remain accessible.

The local issuer is https://auth.localhost:5173/realms/platform, routed through Envoy. API/MCP fetch JWKS over HTTPS on the internal Keycloak service, verifying the project CA. Local certificate tooling includes the auth hostname and service DNS name. The temporary browser smoke test pins the local leaf certificate; ordinary browser trust still uses bin/certs-trust.

Keycloak imports realm.json only when the realm does not already exist. Subsequent realm/client changes require a reviewed admin API migration; a pod restart does not overwrite users or realm configuration. The startup admin is for bootstrap, not application access. Production profiles must override issuer/redirect origins, credentials, environment claim, TLS certificates, sizing/replicas, and local test clients.

Vendor provenance and upgrade notes are in ../../vendor/keycloak/README.md. Database schema upgrades are Keycloak-managed; back up the separate database before upgrades. The Database CR retains data when removed.

Realm import bootstraps a new database; it does not update an existing realm. After changing the three application clients or their token mappers, run `bin/local-identity-sync` with forwarding active. This local command preserves users and passwords; it does not reconcile realm roles or scopes.

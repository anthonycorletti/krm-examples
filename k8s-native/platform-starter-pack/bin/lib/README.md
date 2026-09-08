# Local command implementations

These helpers implement commands in bin/ and are not packaged with the application.
certs.py creates a development CA and a renewable certificate for web.localhost,
api.localhost, mcp.localhost, auth.localhost, and the internal Keycloak service under ignored .local/tls/. It never edits DNS,
the hosts file, or trust stores. bin/certs-trust is the explicit local trust step.

Kubernetes environments supply TLS certificates through their networking resources
and Secrets; they do not run this development CA generator.

local.py implements Colima lifecycle and token commands. experiment_check.py runs live task, export, and API/MCP identity checks. keycloak_sync.py reconciles local application clients without replacing users. web_check.ts powers optional bin/web-smoke using Bun and an installed Chrome (CHROME_BIN overrides the macOS default). It creates and removes its own browser profile, pins only this project's certificate without modifying trust stores, and writes screenshots/Chrome diagnostics to ignored .local/qa/. It exercises Keycloak sign-in, Export now, themes, and mobile layout. This is a UI check, not verification of browser CA trust; use bin/tls-check and the normal browser for that.

local_cleanup.py previews or removes only the starter workload/data inventory on Colima. Full uninstall includes project controllers and CRDs after checking for external dependencies; destructive execution requires both --delete-data and --apply.

# Local command implementations

These helpers implement commands in bin/ and are not packaged with the application.
certs.py creates a development CA and a renewable certificate for web.localhost,
api.localhost, and mcp.localhost under ignored .local/tls/. It never edits DNS,
the hosts file, or trust stores. bin/certs-trust is the explicit local trust step.

Kubernetes environments supply TLS certificates through their networking resources
and Secrets; they do not run this development CA generator.

local.py implements Colima lifecycle and token commands. experiment_check.py runs the live API task walkthrough. web_check.ts powers optional bin/web-smoke using Bun and an installed Chrome (CHROME_BIN overrides the macOS default). It creates and removes its own browser profile, pins only this project's certificate without modifying trust stores, and writes screenshots/Chrome diagnostics to ignored .local/qa/. This is a UI check, not verification of browser CA trust; use bin/tls-check and the normal browser for that.

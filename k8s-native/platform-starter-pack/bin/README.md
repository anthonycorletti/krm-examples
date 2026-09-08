# Commands

Run these scripts from any directory. They resolve the example root themselves. Application commands live here, while dependencies/configuration stay in server/ or web/.

| Command | Purpose |
| --- | --- |
| install | Frozen server and web installs |
| server-install / web-install | Install one frozen lockfile |
| server-add [--dev] name==X.Y.Z | Explicit pinned server dependency change, frozen install, server checks |
| web-smoke | Optional installed-Chrome UI check using Bun; requires local-forward and walkthrough evidence |
| web-ui add COMPONENT | Generate shadcn components with the pinned Bun CLI, then pin dependencies and install frozen |
| web-add [--dev] name@X.Y.Z | Explicit pinned web dependency change, frozen install, web build |
| dev-init | Create ignored local settings and signing secret without overwriting existing settings |
| postgres-start / postgres-stop | Start/stop the labeled local database, retaining its container/data |
| migrate / migration-check | Upgrade Alembic to head / check model drift |
| migration-new "description" | Generate a timestamp_ms_rev_slug migration for review |
| dev-token [--mcp] [--viewer] | Issue a short-lived local token |
| server-dev / mcp-dev / web-dev | Run a development process |
| server-format / web-format | Format the relevant source |
| server-typecheck / web-typecheck | ty / TypeScript checks |
| server-test | Run tests; accepts pytest arguments |
| integration-test | Real Postgres and MCP integration tests in disposable databases |
| agent-demo | Run the Pydantic AI inspection task locally with a deterministic model and real component probes |
| workflow-demo CONTEXT IMAGE@sha256:DIGEST | Submit the installed Argo inspection template, wait for completion, and show task logs |
| server-check / check | Server validation / server validation plus web build |
| export-openapi / generate-client / client-check | Export, generate, and detect contract drift |
| server-build / web-build | Build server container / static web assets |
| vendor-fetch / vendor-check | Restore the minimal pinned vendor set / verify their checksums |
| krm-check | Verify vendor checksums and render vendor, Postgres, Argo controller, and agent workflow resources offline |
| local-secrets-enable | Enable K3s Secret encryption on single-node Colima, restart the control plane, and reencrypt existing Secrets |
| local-secrets-check | Refresh the timestamped control-plane encryption evidence shown in System |
| local-check | Verify vendor checksums and render all Colima compositions |
| local-up | Build images in Colima and start the full local Kubernetes service set |
| local-experiment | Create a project, execute a task and follow-up through Argo, and verify object/report/cache evidence |
| local-components | Read authenticated live component status and evidence from the local API |
| local-status | Show Colima platform pods, services, volumes, and workflows |
| local-forward | Forward Envoy HTTPS to the fixed web/API/MCP localhost ports |
| local-token [--viewer] [--mcp] | Obtain a Keycloak token with the appropriate role and audience |
| local-credentials | Show generated local developer/viewer login credentials |
| local-identity-check | Verify real API/MCP OIDC tokens and audience isolation; show nightly export evidence |
| local-identity-sync | Reconcile local Keycloak client configuration and token mappers without replacing users |
| local-nightly | Invoke the nightly scheduler now; it queues one export per owner |
| local-export | Exercise a real snapshot export, warehouse counts, idempotency, and viewer denial |
| local-cleanup [--delete-data] [--apply] | Preview cleanup; --apply stops workloads, or --delete-data --apply fully uninstalls project resources, controllers, CRDs, and data after dependency checks |
| local-stop | Stop app/data workloads and hibernate Postgres, retaining volumes and controllers |

Cluster provisioning is outside the starter pack. Use an existing cluster; bin/local-* targets Colima. Local postgres-start remains a Docker command with no Kubernetes dependency.

Normal installs never resolve new versions: uv sync --frozen and bun install --frozen-lockfile. Only explicit add commands change dependency metadata/lockfiles, retaining existing compatible resolutions. They do not use upgrade-all. Review lock diffs and run bin/integration-test after dependency changes; a failed check leaves the changes available for correction, not silently accepted. There are no generic lock regeneration commands.

Local HTTPS commands:

- bin/certs creates or renews certificates under ignored .local/tls/. Each dev server calls it automatically.
- bin/certs-trust installs the local CA into the macOS login keychain. This is a one-time local trust step, separate from application startup.
- bin/tls-check verifies the web, API, MCP, and web-to-API proxy using the project CA. Start all three development servers first.

Addresses stay fixed at https://web.localhost:5173, https://api.localhost:8000, and https://mcp.localhost:8001/mcp. No hosts-file or DNS changes are needed. Vite fails if port 5173 is occupied instead of choosing a different port. Restart the dev servers after a certificate renewal. Local TLS tooling lives in bin/lib/ and is not part of the application image.

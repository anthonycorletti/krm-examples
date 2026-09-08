# Local profile

The full local platform uses the existing default Colima profile with Kubernetes enabled and the Docker runtime. It uses the same KRM bases across supported Kubernetes distributions, including Argo, with one Postgres instance and local-path persistent storage. No kind or cloud application service is required.

From the example root:

```sh
./bin/install
./bin/local-check
./bin/local-up
./bin/local-status
./bin/local-forward
```

In another terminal, run `bin/local-credentials` and sign in through Keycloak at https://web.localhost:5173. API and MCP use https://api.localhost:8000 and https://mcp.localhost:8001/mcp. Stop development servers using those ports before starting local-forward. Certificates use the existing local CA; bin/certs-trust installs trust when needed. Forwarding listens only on loopback and stops with Ctrl-C; containers and data continue running.

Every Kubernetes command explicitly targets the colima context. Images are built into the default Colima Docker runtime. local-up preserves existing credentials and PVCs, applies controller resources before custom resources, copies application Postgres credentials between the two namespaces, runs Alembic, and waits for workload readiness. A failed startup leaves resources available for inspection and a later retry; it does not erase data.

`bin/local-stop` stops application/data workloads and hibernates Postgres while retaining volumes, secrets, and controllers. It refuses to stop while workflows are active. Resume with local-up. It does not stop Colima or unrelated workloads.

Services include Postgres, Valkey, SeaweedFS, Argo Workflows, Prometheus, Alertmanager, Jaeger, Perses, OpenSearch, OpenTelemetry Collector, ClickHouse, Envoy Gateway, and cert-manager. Allow sufficient Colima memory and disk for these services and image downloads. This is a single-node development profile; local Secrets encryption at rest is enabled with bin/local-secrets-enable and checked with bin/local-secrets-check, and the OpenSearch endpoint is internal with its security plugin disabled.

The workspace supports projects, queued tasks, follow-up conversations, cancellation, and reports. `bin/local-experiment` exercises two runs through Argo, SeaweedFS, Postgres, and Valkey and leaves the evidence visible in the UI. The default inspector is a deterministic Pydantic AI TestModel calling real platform tools. Model-provider credentials are not configured automatically. Trace IDs are recorded; complete telemetry ingestion, nightly snapshot retention, and production recovery still need validation.

Warehouse includes Export now and a 02:00 UTC nightly schedule. Run bin/local-export for a live export and permission check. The identity route shares port 5173 with web.localhost using auth.localhost.

## Secret encryption

Run `bin/local-secrets-enable` once on the existing single-node Colima cluster. It follows [K3s existing-cluster encryption](https://docs.k3s.io/cli/secrets-encrypt#enable-secrets-encryption-on-an-existing-cluster): initialize encryption, persist the flag in /etc/rancher/k3s/config.yaml.d/90-platform-secrets.yaml, restart K3s, rotate/rewrite existing Secrets, and restart again. API forwarding may need restarting afterward. An interrupted initialization before the first restart can resume with `bin/local-secrets-enable --resume`; inspect the control-plane status before recovery.

`bin/local-secrets-check` reads the control-plane status and publishes a non-secret, timestamped attestation. The System check expires after 24 hours; rerun the command to refresh it. This is an administrator attestation, not continuous datastore inspection. Encryption keys remain root-owned inside Colima; protect the VM disk and backups. Other distributions must supply their own control-plane evidence; local evidence is never fabricated for other clusters.

Use bin/local-cleanup to preview a data-preserving stop, or bin/local-cleanup --delete-data to preview a complete project uninstall, including controllers, CRDs, and PVCs. No cleanup is executed without --apply. See the example README for retained resources.

"""Local platform lifecycle. Always targets the default Colima profile."""

import argparse
import base64
import json
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KUBECTL = ["kubectl", "--context", "colima"]


def command(*args: str, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=capture)
    return result.stdout if capture else ""


def kube(*args: str, capture: bool = False) -> str:
    return command(*KUBECTL, *args, capture=capture)


def apply_object(value: dict, *, sensitive: bool = False) -> None:
    result = subprocess.run(
        [*KUBECTL, "apply", "--server-side", "--field-manager=platform-local", "-f", "-"],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    if result.returncode:
        if sensitive:
            raise RuntimeError("Could not apply local Secret; inspect Kubernetes permissions")
        raise RuntimeError(result.stderr)
    print(result.stdout.strip())


def apply(path: str) -> None:
    kube("apply", "--server-side", "--field-manager=platform-local", "-k", path)


def get(kind: str, name: str, namespace: str) -> dict | None:
    raw = kube("-n", namespace, "get", kind, name, "--ignore-not-found", "-o", "json", capture=True)
    return json.loads(raw) if raw.strip() else None


def secret(name: str, namespace: str, values: dict[str, str], *, encoded: bool = False) -> None:
    apply_object(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name, "namespace": namespace},
            "type": "Opaque",
            "data": values
            if encoded
            else {key: base64.b64encode(value.encode()).decode() for key, value in values.items()},
        },
        sensitive=True,
    )


def bootstrap_secrets() -> None:
    if get("secret", "keycloak-db", "platform-cluster") is None:
        apply_object(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": "keycloak-db", "namespace": "platform-cluster"},
                "type": "kubernetes.io/basic-auth",
                "stringData": {"username": "keycloak", "password": secrets.token_urlsafe(36)},
            },
            sensitive=True,
        )
    if get("secret", "keycloak-users", "platform-cluster") is None:
        secret(
            "keycloak-users",
            "platform-cluster",
            {
                "KC_BOOTSTRAP_ADMIN_PASSWORD": secrets.token_urlsafe(36),
                "KEYCLOAK_DEVELOPER_PASSWORD": secrets.token_urlsafe(24),
                "KEYCLOAK_VIEWER_PASSWORD": secrets.token_urlsafe(24),
            },
        )
    if get("secret", "platform-services", "platform-cluster") is None:
        access, password = secrets.token_hex(16), secrets.token_urlsafe(36)
        secret(
            "platform-services",
            "platform-cluster",
            {
                "valkey-password": secrets.token_urlsafe(36),
                "clickhouse-password": secrets.token_urlsafe(36),
                "s3-access-key": access,
                "s3-secret-key": password,
                "s3.json": json.dumps(
                    {
                        "identities": [
                            {
                                "name": "platform",
                                "credentials": [{"accessKey": access, "secretKey": password}],
                                "actions": ["Admin", "Read", "Write", "List", "Tagging"],
                            }
                        ]
                    }
                ),
            },
        )
    services = get("secret", "platform-services", "platform-cluster")
    if services is None:
        raise RuntimeError("Platform service credentials are missing")
    values = {k: base64.b64decode(v).decode() for k, v in services["data"].items()}
    secret(
        "platform-storage",
        "platform-apps",
        {
            "APP_CLICKHOUSE_PASSWORD": values["clickhouse-password"],
            "APP_OBJECT_ACCESS_KEY": values["s3-access-key"],
            "APP_OBJECT_SECRET_KEY": values["s3-secret-key"],
            "APP_VALKEY_URL": f"redis://:{values['valkey-password']}@valkey.platform-cluster.svc.cluster.local:6379/0",
        },
    )
    command(str(ROOT / "bin/certs"))
    cert = ROOT / ".local/tls"
    secret("platform-oidc-ca", "platform-apps", {"ca.crt": (cert / "ca.crt").read_text()})
    apply_object(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "kubernetes.io/tls",
            "metadata": {"name": "platform-local-tls", "namespace": "platform-cluster"},
            "data": {
                key: base64.b64encode((cert / file).read_bytes()).decode()
                for key, file in {"tls.crt": "localhost.crt", "tls.key": "localhost.key"}.items()
            },
        },
        sensitive=True,
    )


def preflight() -> None:
    nodes = json.loads(kube("get", "nodes", "-o", "json", capture=True))["items"]
    if not nodes or any(
        not n["status"]["nodeInfo"]["containerRuntimeVersion"].startswith("docker://")
        for n in nodes
    ):
        raise RuntimeError(
            "This local image workflow requires Colima's Docker runtime with Kubernetes enabled"
        )
    if not any(
        c["type"] == "Ready" and c["status"] == "True"
        for n in nodes
        for c in n["status"]["conditions"]
    ):
        raise RuntimeError("Colima Kubernetes is not Ready")
    kube("get", "storageclass", "local-path")


def up() -> None:
    preflight()
    socket = Path.home() / ".config/colima/default/docker.sock"
    if not socket.exists():
        raise RuntimeError("The default Colima Docker socket is missing")
    for section in ("server", "web"):
        command(
            "docker",
            "--host",
            f"unix://{socket}",
            "build",
            "-t",
            f"platform-starter-{section}:local",
            section,
        )
    kube("apply", "-f", "k8s/base/cluster/namespaces.yaml")
    bootstrap_secrets()
    for controller, deployment in (
        ("cloudnative-pg", "cnpg-controller-manager"),
        ("cert-manager", "cert-manager"),
        ("envoy-gateway", "envoy-gateway"),
    ):
        apply(f"k8s/base/cluster/{controller}")
    for deployment in ("cnpg-controller-manager", "cert-manager", "envoy-gateway"):
        kube(
            "-n",
            "platform-cluster",
            "rollout",
            "status",
            f"deployment/{deployment}",
            "--timeout=300s",
        )
    apply("k8s/profiles/local/services")
    kube(
        "-n",
        "platform-cluster",
        "annotate",
        "cluster/platform-postgres",
        "cnpg.io/hibernation=off",
        "--overwrite",
    )
    kube(
        "-n",
        "platform-cluster",
        "wait",
        "cluster/platform-postgres",
        "--for=condition=Ready",
        "--timeout=600s",
    )
    for name in ("platform-postgres-app", "platform-postgres-ca"):
        source = get("secret", name, "platform-cluster")
        if source is None:
            raise RuntimeError(f"Postgres has not created {name}")
        secret(name, "platform-apps", source["data"], encoded=True)
    apply("k8s/profiles/local/routing")
    apply("k8s/profiles/local/apps")
    api = get("deployment", "platform-api", "platform-apps")
    if api and api.get("spec", {}).get("replicas") == 0:
        kube(
            "-n",
            "platform-apps",
            "scale",
            "deployment/platform-api",
            "--replicas=1",
            "--field-manager=platform-local",
        )
    old_job = get("job", "platform-migrate", "platform-apps")
    if old_job:
        if old_job.get("status", {}).get("active", 0):
            raise RuntimeError(
                "A migration is already running; wait for it before rerunning local-up"
            )
        kube("-n", "platform-apps", "delete", "job", "platform-migrate", "--wait=true")
    apply("k8s/profiles/local/migrations")
    kube(
        "-n",
        "platform-apps",
        "wait",
        "job/platform-migrate",
        "--for=condition=Complete",
        "--timeout=300s",
    )
    apply("k8s/profiles/local/workflows")
    for ns in ("platform-cluster", "platform-apps"):
        if ns == "platform-apps":
            kube("-n", ns, "rollout", "restart", "deployment")
        kube("-n", ns, "rollout", "status", "deployment", "--timeout=600s")
        kube("-n", ns, "rollout", "status", "statefulset", "--timeout=600s")
    print("Containers are ready. Run bin/local-forward, then bin/local-credentials to sign in.")
    print(
        "Open the experimentation workspace with bin/local-forward and bin/local-credentials. "
        "Run bin/local-experiment to exercise a task and follow-up through Argo."
    )


def forward() -> None:
    raw = kube(
        "-n",
        "platform-cluster",
        "get",
        "service",
        "-l",
        "gateway.envoyproxy.io/owning-gateway-name=platform",
        "-o",
        "json",
        capture=True,
    )
    services = json.loads(raw)["items"]
    if len(services) != 1:
        raise RuntimeError("Expected one Envoy service for the local platform Gateway")
    name = services[0]["metadata"]["name"]
    print(
        "Web https://web.localhost:5173 | API https://api.localhost:8000 | MCP https://mcp.localhost:8001/mcp",
        flush=True,
    )
    command(
        *KUBECTL,
        "-n",
        "platform-cluster",
        "port-forward",
        "--address=127.0.0.1",
        f"service/{name}",
        "5173:443",
        "8000:443",
        "8001:443",
    )


def status() -> None:
    for namespace in ("platform-apps", "platform-cluster"):
        kube("-n", namespace, "get", "pods,services,pvc")
    kube("-n", "platform-apps", "get", "workflows")


def workflow(template="platform-inspect") -> None:
    result = subprocess.run(
        [*KUBECTL, "-n", "platform-apps", "create", "-f", "-", "-o", "json"],
        input=json.dumps(
            {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {"generateName": template + "-"},
                "spec": {"workflowTemplateRef": {"name": template}},
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    name = json.loads(result.stdout)["metadata"]["name"]
    print(f"Submitted {name}", flush=True)
    kube(
        "-n",
        "platform-apps",
        "wait",
        f"workflow/{name}",
        "--for=jsonpath={.status.finishedAt}",
        "--timeout=330s",
    )
    kube(
        "-n",
        "platform-apps",
        "logs",
        "-l",
        f"workflows.argoproj.io/workflow={name}",
        "-c",
        "main",
        "--tail=-1",
    )
    value = get("workflow", name, "platform-apps")
    if value is None or value.get("status", {}).get("phase") != "Succeeded":
        raise RuntimeError(f"Workflow {name} did not succeed; inspect bin/local-status")
    print(f"Workflow {name} succeeded")


def stop() -> None:
    if get("cronworkflow", "platform-nightly-export", "platform-apps"):
        kube(
            "-n",
            "platform-apps",
            "patch",
            "cronworkflow",
            "platform-nightly-export",
            "--type=merge",
            "--field-manager=platform-local",
            "-p",
            '{"spec":{"suspend":true}}',
        )
    active = json.loads(
        kube("-n", "platform-apps", "get", "workflows", "-o", "json", capture=True)
    )["items"]
    if any(
        w.get("status", {}).get("phase") not in ("Succeeded", "Failed", "Error") for w in active
    ):
        raise RuntimeError(
            "Finish or explicitly terminate active workflows before stopping platform services"
        )
    kube("-n", "platform-apps", "scale", "deployment", "--all", "--replicas=0")
    kube(
        "-n",
        "platform-cluster",
        "annotate",
        "cluster/platform-postgres",
        "cnpg.io/hibernation=on",
        "--overwrite",
    )
    kube(
        "-n",
        "platform-cluster",
        "wait",
        "cluster/platform-postgres",
        "--for=condition=cnpg.io/hibernation",
        "--timeout=300s",
    )
    kube("-n", "platform-cluster", "scale", "statefulset", "--all", "--replicas=0")
    # Leave controllers running; Envoy's generated proxy is controller-managed.
    for deployment in ("valkey", "jaeger", "otel-collector", "keycloak"):
        kube("-n", "platform-cluster", "scale", f"deployment/{deployment}", "--replicas=0")
    print(
        "Application and data services stopped; PVCs, secrets, and controllers retained. "
        "Resume with bin/local-up."
    )


def token() -> None:
    import ssl

    import httpx

    role = "viewer" if "--viewer" in sys.argv else "developer"
    value = get("secret", "keycloak-users", "platform-cluster")
    if value is None:
        raise RuntimeError("Run bin/local-up first")
    password = base64.b64decode(value["data"][f"KEYCLOAK_{role.upper()}_PASSWORD"]).decode()
    context = ssl.create_default_context(cafile=str(ROOT / ".local/tls/ca.crt"))
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    response = httpx.post(
        "https://127.0.0.1:5173/realms/platform/protocol/openid-connect/token",
        headers={"Host": "auth.localhost"},
        verify=context,
        data={
            "grant_type": "password",
            "client_id": "platform-mcp-cli" if "--mcp" in sys.argv else "platform-cli",
            "username": role,
            "password": password,
            "scope": "openid platform:read platform:verify",
        },
        timeout=15,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Local Keycloak sign-in failed (HTTP {response.status_code})")
    print(response.json()["access_token"])


def credentials() -> None:
    value = get("secret", "keycloak-users", "platform-cluster")
    if value is None:
        raise RuntimeError("Run bin/local-up first")
    print("Sign in at https://web.localhost:5173")
    for role in ("developer", "viewer"):
        password = base64.b64decode(value["data"][f"KEYCLOAK_{role.upper()}_PASSWORD"]).decode()
        print(f"{role}: {password}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("up", "status", "forward", "token", "stop", "workflow", "credentials", "nightly"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--mcp", action="store_true")
    args = parser.parse_args()
    try:
        {
            "up": up,
            "status": status,
            "forward": forward,
            "token": token,
            "credentials": credentials,
            "stop": stop,
            "workflow": workflow,
            "nightly": lambda: workflow("platform-export-schedule"),
        }[args.action]()
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        sys.exit(str(exc))

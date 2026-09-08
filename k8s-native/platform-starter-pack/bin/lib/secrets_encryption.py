"""Enable and attest Secret encryption on the existing single-server Colima cluster."""

import json
import sys
import time
from datetime import UTC, datetime

from local import apply_object, command, get, kube


def vm(*args):
    return command("colima", "ssh", "--", *args, capture=True)


def main():
    config = json.loads(kube("config", "view", "--minify", "-o", "json", capture=True))
    port = config["clusters"][0]["cluster"]["server"].rsplit(":", 1)[1]
    server = f"https://127.0.0.1:{int(port)}"

    def encrypt(action):
        return vm("sudo", "k3s", "secrets-encrypt", action, "--server", server)

    def wait_ready():
        for _ in range(120):
            try:
                status = encrypt("status")
                return status
            except Exception:
                time.sleep(2)
        raise RuntimeError("K3s did not return after restart")

    resume = "--resume" in sys.argv
    status = "" if resume else encrypt("status")
    if "--enable" in sys.argv and "Encryption Status: Enabled" not in status:
        nodes = json.loads(kube("get", "nodes", "-o", "json", capture=True))["items"]
        if len(nodes) != 1:
            raise RuntimeError("This command supports only single-node Colima")
        if not resume:
            encrypt("enable")
        vm("sudo", "mkdir", "-p", "/etc/rancher/k3s/config.yaml.d")
        vm(
            "sudo",
            "python3",
            "-c",
            "from pathlib import Path; "
            "Path('/etc/rancher/k3s/config.yaml.d/90-platform-secrets.yaml')"
            ".write_text('secrets-encryption: true\\n')",
        )
        vm("sudo", "systemctl", "restart", "k3s")
        wait_ready()
        print("K3s restarted with encryption configured; rewriting existing Secrets.", flush=True)
        encrypt("rotate-keys")
        for _ in range(180):
            status = encrypt("status")
            if "reencrypt_finished" in status:
                break
            time.sleep(2)
        else:
            raise RuntimeError(
                "Secret reencryption did not finish; inspect k3s secrets-encrypt status"
            )
        vm("sudo", "systemctl", "restart", "k3s")
        status = wait_ready()
    if "Encryption Status: Enabled" not in status or "All hashes match" not in status:
        raise RuntimeError("Encryption is not enabled and consistent; run bin/local-secrets-enable")
    if "reencrypt_finished" not in status:
        raise RuntimeError("Existing Secrets have not completed reencryption")
    cluster = get("namespace", "kube-system", "default")
    assert cluster
    apply_object(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "platform-encryption-evidence", "namespace": "platform-apps"},
            "data": {
                "provider": "k3s",
                "cluster_uid": cluster["metadata"]["uid"],
                "verified_at": datetime.now(UTC).isoformat(),
                "status": status.strip(),
            },
        }
    )
    print(status.strip())
    print("Published control-plane evidence; no Secret values or encryption keys were read.")


if __name__ == "__main__":
    main()

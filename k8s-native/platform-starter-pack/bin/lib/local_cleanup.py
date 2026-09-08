"""Preview or fully uninstall the starter pack from the existing Colima cluster."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import yaml
from local import get, kube, stop

NAMESPACES = {"platform-apps", "platform-cluster"}
STAGES = [
    *(f"k8s/base/cluster/{name}" for name in ("cloudnative-pg", "cert-manager", "envoy-gateway")),
    *(
        f"k8s/profiles/local/{name}"
        for name in ("services", "routing", "apps", "migrations", "workflows")
    ),
]
GLOBAL_TYPES = (
    "customresourcedefinitions",
    "clusterroles",
    "clusterrolebindings",
    "mutatingwebhookconfigurations",
    "validatingwebhookconfigurations",
    "validatingadmissionpolicies",
    "validatingadmissionpolicybindings",
    "priorityclasses",
)


def reference(item):
    group = item["apiVersion"].split("/")[0] if "/" in item["apiVersion"] else ""
    return item["kind"] + (f".{group}" if group else "")


def identity(item):
    return reference(item), item["metadata"].get("namespace", ""), item["metadata"]["name"]


def project_managed(item):
    return any(
        f.get("manager") == "platform-local" for f in item["metadata"].get("managedFields", [])
    )


def resources():
    result = {}
    for stage in STAGES:
        for item in yaml.safe_load_all(kube("kustomize", stage, capture=True)):
            if item:
                result[identity(item)] = item
    return list(result.values())


def listing(kind, *args):
    return json.loads(
        kube("get", kind, *args, "-o", "json", "--show-managed-fields", capture=True)
    )["items"]


def reject_foreign_custom_resources(items, expected):
    for item in items:
        namespace = item["metadata"].get("namespace")
        if namespace in NAMESPACES:
            continue
        if not namespace and (identity(item) in expected or project_managed(item)):
            continue
        raise RuntimeError(f"Shared CRD dependency outside this project: {identity(item)}")


def plan():
    inventory = resources()
    expected = {identity(item) for item in inventory}
    global_live = listing(",".join(GLOBAL_TYPES))
    globals_to_delete = [
        item for item in global_live if identity(item) in expected or project_managed(item)
    ]
    for item in globals_to_delete:
        if not project_managed(item):
            raise RuntimeError(f"Cannot establish project ownership of {identity(item)}")
    for ns in NAMESPACES:
        namespace = get("namespace", ns, "default")
        if (
            namespace
            and namespace["metadata"].get("labels", {}).get("app.kubernetes.io/part-of")
            != "platform-starter-pack"
        ):
            raise RuntimeError(f"Namespace {ns} is not labeled as belonging to this project")
    crds = [item for item in globals_to_delete if item["kind"] == "CustomResourceDefinition"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(
            pool.map(lambda crd: listing(crd["metadata"]["name"], "--all-namespaces"), crds)
        )
    custom = [item for batch in batches for item in batch]
    reject_foreign_custom_resources(custom, expected)
    roles = {
        item["metadata"]["name"] for item in globals_to_delete if item["kind"] == "ClusterRole"
    }
    bindings = listing("rolebindings", "--all-namespaces") + [
        item for item in global_live if item["kind"] == "ClusterRoleBinding"
    ]
    own_bindings = {identity(item) for item in globals_to_delete}
    for item in bindings:
        if (
            item.get("roleRef", {}).get("kind") == "ClusterRole"
            and item["roleRef"]["name"] in roles
            and item["metadata"].get("namespace") not in NAMESPACES
            and identity(item) not in own_bindings
        ):
            raise RuntimeError(f"Shared controller RBAC dependency: {identity(item)}")
    volumes = [
        item
        for item in listing("pv")
        if item.get("spec", {}).get("claimRef", {}).get("namespace") in NAMESPACES
    ]
    for volume in volumes:
        if volume["spec"].get("persistentVolumeReclaimPolicy") != "Delete":
            raise RuntimeError(
                f"PV {volume['metadata']['name']} retains backing storage; "
                "resolve its reclaim policy before full cleanup"
            )
    return globals_to_delete, custom, volumes


def delete(item):
    namespace = item["metadata"].get("namespace")
    kube(
        *(["-n", namespace] if namespace else []),
        "delete",
        reference(item),
        item["metadata"]["name"],
        "--ignore-not-found",
        "--wait=true",
        "--timeout=300s",
    )


def uninstall(globals_to_delete, custom, volumes):
    # Quiesce producers; controllers remain alive to process custom-resource finalizers.
    if get("namespace", "platform-apps", "default"):
        for item in listing("cronworkflows", "-n", "platform-apps"):
            kube(
                "-n",
                "platform-apps",
                "patch",
                "cronworkflow",
                item["metadata"]["name"],
                "--type=merge",
                "-p",
                '{"spec":{"suspend":true}}',
            )
        kube("-n", "platform-apps", "scale", "deployment", "--all", "--replicas=0")
    order = {
        "CronWorkflow": 0,
        "Workflow": 1,
        "Database": 2,
        "Cluster": 3,
        "Gateway": 4,
        "GatewayClass": 6,
    }
    for item in sorted(custom, key=lambda i: order.get(i["kind"], 5)):
        delete(item)
    # Remove admission callbacks before removing their serving pods.
    admission = [
        i
        for i in globals_to_delete
        if "WebhookConfiguration" in i["kind"] or "AdmissionPolicy" in i["kind"]
    ]
    for item in admission:
        delete(item)
    for ns in ("platform-apps", "platform-cluster"):
        kube("delete", "namespace", ns, "--ignore-not-found", "--wait=true", "--timeout=300s")
    # Keep the cluster's storage provisioner running until project volumes are reclaimed.
    for item in volumes:
        kube("wait", "--for=delete", f"pv/{item['metadata']['name']}", "--timeout=300s")
    for item in globals_to_delete:
        if item not in admission:
            delete(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Execute the printed cleanup mode")
    parser.add_argument(
        "--delete-data",
        action="store_true",
        help="Uninstall all project cluster resources and data",
    )
    args = parser.parse_args()
    if not args.delete_data:
        print("Context: colima. Stop app/data workloads; retain resources and data.")
        if args.apply:
            stop()
        else:
            print("Preview only. Run bin/local-cleanup --apply to stop workloads.")
        return
    globals_to_delete, custom, volumes = plan()
    print(
        "Context: colima. FULL UNINSTALL, including project controllers, "
        "CRDs, credentials, and data."
    )
    for ns in sorted(NAMESPACES):
        print(f"DELETE Namespace/{ns} and ALL contents, including PVCs and Secrets")
    for item in [*custom, *globals_to_delete, *volumes]:
        print("DELETE " + "/".join(identity(item)))
    print(
        "Colima/K3s, storage provisioner, cluster encryption, "
        "local files, and Docker images remain."
    )
    if not args.apply:
        print("Preview only. Execute with bin/local-cleanup --delete-data --apply")
        return
    uninstall(globals_to_delete, custom, volumes)
    print("Project uninstalled from Kubernetes. Run bin/local-up for a fresh installation.")


if __name__ == "__main__":
    main()

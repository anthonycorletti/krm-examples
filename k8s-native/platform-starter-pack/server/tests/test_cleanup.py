"""Full cleanup must reject shared dependencies and preserve finalizer ordering."""

import importlib
from pathlib import Path

import pytest


def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "bin/lib"))
    return importlib.import_module("local_cleanup")


def resource(kind, name, namespace=None):
    metadata = {"name": name}
    if namespace:
        metadata["namespace"] = namespace
    return {"apiVersion": "example.test/v1", "kind": kind, "metadata": metadata}


def test_cleanup_rejects_foreign_custom_resources(monkeypatch):
    cleanup = helper(monkeypatch)
    own = resource("Cluster", "platform-postgres", "platform-cluster")
    foreign = resource("Cluster", "customer-db", "other-team")
    cleanup.reject_foreign_custom_resources([own], set())
    with pytest.raises(RuntimeError, match="Shared CRD dependency"):
        cleanup.reject_foreign_custom_resources([own, foreign], set())
    gateway = resource("GatewayClass", "other-ingress")
    with pytest.raises(RuntimeError, match="Shared CRD dependency"):
        cleanup.reject_foreign_custom_resources([gateway], set())
    cleanup.reject_foreign_custom_resources([gateway], {cleanup.identity(gateway)})


def test_cleanup_preview_never_mutates(monkeypatch, capsys):
    cleanup = helper(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted a cluster mutation")

    monkeypatch.setattr(cleanup, "kube", forbidden)
    monkeypatch.setattr(cleanup, "stop", forbidden)
    monkeypatch.setattr(cleanup, "uninstall", forbidden)
    monkeypatch.setattr(cleanup, "plan", lambda: ([], [], []))
    for args in (["cleanup"], ["cleanup", "--delete-data"]):
        monkeypatch.setattr("sys.argv", args)
        cleanup.main()
        assert "Preview only" in capsys.readouterr().out


def test_full_cleanup_keeps_controllers_until_custom_resources_are_gone(monkeypatch):
    cleanup = helper(monkeypatch)
    calls = []
    monkeypatch.setattr(cleanup, "kube", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(cleanup, "get", lambda *args: None)
    crd = resource("CustomResourceDefinition", "clusters.example.test")
    webhook = resource("ValidatingWebhookConfiguration", "platform-webhook")
    cluster = resource("Cluster", "platform-postgres", "platform-cluster")
    database = resource("Database", "platform-keycloak", "platform-cluster")
    volume = resource("PersistentVolume", "project-data")
    cleanup.uninstall([crd, webhook], [cluster, database], [volume])
    actions = [" ".join(call) for call in calls]

    def index(fragment):
        return next(i for i, action in enumerate(actions) if fragment in action)

    assert index("Database.example.test") < index("Cluster.example.test")
    assert index("Cluster.example.test") < index("ValidatingWebhookConfiguration.example.test")
    assert index("ValidatingWebhookConfiguration.example.test") < index(
        "delete namespace platform-cluster"
    )
    assert index("delete namespace platform-cluster") < index("wait --for=delete pv/project-data")
    assert index("wait --for=delete pv/project-data") < index(
        "CustomResourceDefinition.example.test"
    )

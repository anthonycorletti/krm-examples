"""Reconcile only the three local application clients; preserve users and passwords."""

import base64
import json
import ssl

import httpx
from local import ROOT, get


def main():
    value = get("secret", "keycloak-users", "platform-cluster")
    if value is None:
        raise RuntimeError("Run bin/local-up first")
    password = base64.b64decode(value["data"]["KC_BOOTSTRAP_ADMIN_PASSWORD"]).decode()
    context = ssl.create_default_context(cafile=str(ROOT / ".local/tls/ca.crt"))
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    with httpx.Client(
        base_url="https://127.0.0.1:5173",
        verify=context,
        headers={"Host": "auth.localhost"},
        timeout=30,
    ) as client:
        response = client.post(
            "/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": "admin",
                "password": password,
            },
        )
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        realm = json.loads(
            (ROOT / "k8s/base/cluster/keycloak/realm.json")
            .read_text()
            .replace("${APP_ENV}", "local")
        )
        for desired in realm["clients"]:
            response = client.get(
                "/admin/realms/platform/clients", params={"clientId": desired["clientId"]}
            )
            response.raise_for_status()
            existing = response.json()
            if len(existing) != 1:
                raise RuntimeError("Expected an existing local application client")
            path = f"/admin/realms/platform/clients/{existing[0]['id']}"
            response = client.put(path, json={**desired, "id": existing[0]["id"]})
            response.raise_for_status()
            response = client.get(path + "/protocol-mappers/models")
            response.raise_for_status()
            mappers = {m["name"]: m for m in response.json()}
            for mapper in desired["protocolMappers"]:
                old = mappers.get(mapper["name"])
                if old:
                    response = client.put(
                        path + f"/protocol-mappers/models/{old['id']}",
                        json={**mapper, "id": old["id"]},
                    )
                else:
                    response = client.post(path + "/protocol-mappers/models", json=mapper)
                response.raise_for_status()
            print(f"Reconciled {desired['clientId']}")


if __name__ == "__main__":
    main()

# Keycloak runtime

Pinned release: 26.7.3. Source: https://github.com/keycloak/keycloak/tree/26.7.3. License: Apache-2.0 (https://github.com/keycloak/keycloak/blob/26.7.3/LICENSE.txt).

Image: quay.io/keycloak/keycloak:26.7.3@sha256:ff4257d0d64efbe99ed1ddfaf07765cc3c36dc7518bf8324d41961327f441c54. The multi-platform digest was checked with docker buildx imagetools inspect; the runtime inventory is ../runtimes/images.json. No upstream manifests or operator are copied. Our Deployment, database resource, and realm configuration live in ../../cluster/keycloak/.

For upgrades, review upstream release/security notes and database migration guidance, update the pinned image/inventory together, preserve database backups, then exercise browser PKCE, developer/viewer permissions, API/MCP audience rejection, and export workflows. Realm bootstrap imports do not reconcile existing realms; evolve live configuration through reviewed admin API migrations.

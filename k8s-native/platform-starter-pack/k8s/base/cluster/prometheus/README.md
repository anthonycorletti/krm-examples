# prometheus

Scrapes API request metrics and configured platform endpoints, and evaluates alerts. Kustomize generates a ConfigMap with a content hash, so configuration updates change the pod template and trigger a rollout. Review target health after configuration changes; complete dashboards and alert delivery remain work ahead.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the local K3s composition.

# valkey

Disposable, authenticated progress entries and notification fan-out. The app uses a five-minute TTL; Postgres remains the durable run queue. Credentials come from platform-services and the application platform-storage Secret.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the local K3s composition.

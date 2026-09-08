# seaweedfs

S3-compatible storage for immutable messages, model history, and reports. The local deployment uses a single server and persistent storage. Application credentials come from Kubernetes Secrets. Orphan cleanup, backup/restore, and multi-node recovery remain to be validated.

Runtime versions and image digests are recorded in [the vendor inventory](../../vendor/README.md). Apply through the selected profile; `bin/local-check` renders the local K3s composition.

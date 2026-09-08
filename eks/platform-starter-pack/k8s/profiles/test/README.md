# Test profile

The first Kubernetes deployment test targets EKS, using an explicit cluster/context and an isolated test namespace. Cluster selection and access are pending; no EKS deployment has been validated.

First exercise: install the pinned CloudNativePG operator, provision Postgres on encrypted EBS storage, run Alembic as a Job with an immutable server image, and verify database connectivity, migration head, and persisted evidence. Authentication remains a separate prerequisite for exposing the application remotely; development authentication must not be published as a cloud login solution.

Before implementation, establish the AWS account/region, cluster version, compute architecture, image registry access, and storage provisioner. EKS Auto Mode uses ebs.csi.eks.amazonaws.com; the standard EBS CSI driver uses ebs.csi.aws.com. Choose the matching storage class after inspecting the target. See [AWS storage documentation](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html).

Cleanup must remove only resources owned by the test run, with an explicit data-retention decision. Shared operators and the EKS cluster are outside namespace cleanup.

Current bin/integration-test uses disposable Postgres databases and real HTTP MCP without a Kubernetes cluster. CI runs that suite until this profile is implemented.

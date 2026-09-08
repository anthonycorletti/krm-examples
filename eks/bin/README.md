# EKS commands

Small shell scripts using your locally installed AWS CLI v2 and its normal credentials/AWS_PROFILE. Run from any directory. Cluster name defaults to krm-examples; pass one name argument to override it. Set AWS_REGION explicitly.

| Command | Purpose |
| --- | --- |
| create-cluster [NAME] | Submit creation of an EKS Auto Mode cluster |
| cluster-status [NAME] | Inspect cluster status, version, and health |
| cluster-kubeconfig [NAME] | Write eks/.local/NAME.kubeconfig using the local AWS CLI |
| delete-cluster [NAME] | Submit cluster deletion |
| check | Check shell syntax and command forwarding without AWS calls |

Like gke/bin/create-cluster, this creates a managed cluster. EKS additionally requires existing subnets and IAM roles. Supply an Auto Mode cluster role and node role with the required trust policies and permissions, plus subnets in at least two availability zones with suitable routing and available IP addresses. These scripts do not create or delete the VPC, subnets, or IAM roles. See [AWS's Auto Mode prerequisites](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster-auto.html).

From eks/:

```sh
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
export EKS_CLUSTER_ROLE_ARN=arn:aws:iam::123456789012:role/your-auto-cluster-role
export EKS_NODE_ROLE_ARN=arn:aws:iam::123456789012:role/your-auto-node-role
export EKS_SUBNET_IDS=subnet-replace1,subnet-replace2
export EKS_API_CIDR=203.0.113.10/32 # Replace with your administrator egress CIDR.
./bin/create-cluster
./bin/cluster-status
# After status is ACTIVE:
./bin/cluster-kubeconfig
```

Replace the example account, roles, subnet IDs, and CIDR. Kubernetes defaults to 1.36; EKS_VERSION overrides it. The creating IAM principal receives cluster administrator access. Auto Mode manages compute, networking, load balancing, and block storage; workload readiness and the platform's storage-class configuration still need verification. Creation returns before the cluster is active. These commands have not yet been exercised against a live AWS account.

The generated kubeconfig uses your local aws executable for authentication and leaves the default kubeconfig unchanged. Select the generated file explicitly for deployment commands.

Before running bin/delete-cluster, remove workloads and Kubernetes-managed load balancers while their controllers are running, and decide data retention for PVCs and backups. Deletion is asynchronous and does not delete the supplied VPC/subnets/roles or guarantee that retained volumes are removed. AWS resources incur charges while present.

AWS-specific setup stays here. The platform uses portable Kubernetes workloads, PVCs, Secrets, and Services; its application functionality does not require AWS SDK calls or managed application services.

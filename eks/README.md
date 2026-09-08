# EKS examples

[bin/](bin/README.md) contains small cluster creation, status, kubeconfig, and deletion scripts using your local AWS CLI. The scripts target EKS Auto Mode with existing subnets and IAM roles.

[platform-starter-pack/](platform-starter-pack/README.md) contains the portable Kubernetes platform. EKS is the first deployment-test target. AWS supplies the hosting infrastructure; platform functionality remains in Kubernetes-native services. Local application development uses Docker Postgres without requiring Kubernetes.

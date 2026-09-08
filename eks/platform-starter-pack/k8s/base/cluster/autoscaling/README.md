# Autoscaling

The API HPA uses existing cluster CPU metrics, targets 70% utilization of requested CPU, and allows one to three replicas. No additional Metrics Server or KEDA is installed. The catalog checks ScalingActive; this does not substitute for a load test. Workloads must declare CPU requests. Scaling policies belong to cluster configuration even when their targets are in platform-apps.

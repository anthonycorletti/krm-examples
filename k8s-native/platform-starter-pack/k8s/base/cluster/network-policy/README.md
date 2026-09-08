# Service access

Platform services accept connections from platform-cluster and the API, MCP, and worker pods in platform-apps. This requires a network-policy enforcing CNI. It is an ingress policy, not a complete egress policy or substitute for service authentication.

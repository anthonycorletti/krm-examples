# Kubernetes resources

The layout is base/{apps,cluster,vendor} with profiles/{local,test,preview,production} as overlays.

Developer-owned workloads belong in base/apps/ and platform-apps. Postgres, platform services, operators, networking, certificates, namespace definitions, and scaling belong in base/cluster/ and platform-cluster. Unmodified upstream copies belong in base/vendor/.

The local K3s cluster is the currently validated deployment target. The full local platform uses Kubernetes; native application development can use separate Docker Postgres. No cluster is installed by rendering these resources. Apply in stages: namespaces, controllers/CRDs and readiness checks, custom resources, migrations, applications, and verification.

The local stack has passed staged installation and readiness checks on the local K3s cluster. Remote profiles remain pending. Environment profiles must preserve the two-namespace separation; concurrent preview environments need an explicit isolation design.

# Kubernetes resources

The layout is base/{apps,cluster,vendor} with profiles/{local,test,preview,production} as overlays.

Developer-owned workloads belong in base/apps/ and platform-apps. Postgres, platform services, operators, networking, certificates, namespace definitions, and scaling belong in base/cluster/ and platform-cluster. Unmodified upstream copies belong in base/vendor/.

EKS is the first deployment-test target. Local application development uses Docker Postgres without Kubernetes. No cluster is installed by rendering these resources. Apply in stages: namespaces, controllers/CRDs and readiness checks, custom resources, migrations, applications, and verification.

The minimal vendor inputs and Postgres compositions are prepared; the complete EKS profile remains pending. Do not describe it as deployable until staged installation and readiness tests pass. Environment profiles must preserve the two-namespace separation; concurrent preview environments need an explicit isolation design.

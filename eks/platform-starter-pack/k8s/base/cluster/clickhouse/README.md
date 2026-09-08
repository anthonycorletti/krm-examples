# ClickHouse warehouse

Stores validated metadata snapshots exported through Argo and SeaweedFS Parquet objects. Each export has its own table; application queries select the newest completed snapshot belonging to the caller. Nightly exports run at 02:00 UTC, and Export now runs the same worker. CDC is deferred. See server/app/exports/README.md in the example for snapshot and retention limits.

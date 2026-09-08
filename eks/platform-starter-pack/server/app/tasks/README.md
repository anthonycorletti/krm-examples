# Tasks and conversations

Create a task inside a project with a title, agent, and instructions. Each instruction creates a durable Run and a Message metadata row. The message body is uploaded to an immutable SeaweedFS object before SQL commits the references. Failed SQL transactions can leave unreferenced objects; automated orphan cleanup is not implemented yet.

Argo runs app.worker with the run ID. A dispatcher reconciles durable run state with deterministic workflow names, retries missing submissions, and observes terminal workflow failures. One run per task is selected at a time; other tasks can run concurrently. Follow-up instructions queue another run and reuse Pydantic AI history from object storage. Cancellation is recorded in SQL and observed by workers and the dispatcher.

Reports and Pydantic AI history stay in object storage. SQL holds IDs, ownership, timestamps, run status, token counts, workflow names, trace IDs, and artifact references. Valkey stores expiring progress and notifications; SQL remains authoritative if Valkey fails. Reads and mutations check project ownership, environment, scopes, and soft deletion.

The inspector uses a deterministic TestModel and real service probes. Research and analysis agents use APP_AGENT_MODEL when configured; they are disabled in the UI otherwise. This is trusted tool execution, not an arbitrary-code sandbox. Runs have bounded requests, token budgets, deadlines, and infrastructure retries. Provider requests are not guaranteed exactly once after process failure.

The initial API returns up to 100 tasks and 200 messages/runs/events/artifacts per task. Bodies and histories have a 1 MiB object limit. Pagination, object retention cleanup, provider credential management UI, and general PostgreSQL CDC remain further work.

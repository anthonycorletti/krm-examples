# Agents

Pydantic AI owns model interaction and tools; Argo owns long-running execution. No agent-specific infrastructure vendor or untrusted-code runtime is installed.

service.py defines the async platform inspection tool with an explicit read-only Principal. execution.py loads task instructions/history from SeaweedFS, runs the agent, stores the response and history in SeaweedFS, and commits message/artifact metadata in Postgres. It records execution events, workflow/trace IDs, and token usage. Valkey progress is optional and expires. Instrumentation excludes prompt and response content from spans.

app.worker dispatch reconciles SQL runs with Argo workflows. app.worker run RUN_ID executes a claimed run with a five-minute application deadline. The deterministic inspector needs no provider key. Research/analysis agents require APP_AGENT_MODEL and the appropriate SDK credential supplied to workflow pods.

bin/local-experiment exercises a project, an initial task, a follow-up, persisted messages, artifacts, Argo workflow names, trace IDs, and Valkey progress against the running local platform. It leaves a walkthrough project for inspection. Unit/integration tests additionally check ownership, soft deletion, storage failure, and concurrent dispatcher claims.

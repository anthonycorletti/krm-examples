import { useEffect, useState } from "react";
import {
  Activity,
  Database,
  ArrowLeft,
  ArrowUpRight,
  Folder,
  Plus,
  Square,
  Terminal,
  RefreshCw,
  Trash2,
} from "lucide-react";
import {
  Form,
  Link,
  redirect,
  useFetcher,
  useLoaderData,
  useRevalidator,
  type ActionFunctionArgs,
  type LoaderFunctionArgs,
} from "react-router";
import {
  listProjects,
  createProject,
  deleteProject,
  listTasks,
  createTask,
  readTask,
  sendMessage,
  cancelTask,
  deleteTask,
  listAgents,
  listComponents,
  readIdentity,
  readArtifact,
  listExports,
  createExport,
  readWarehouse,
} from "../client/sdk.gen";
import { hasToken } from "../session";
import { disconnect } from "../oidc";
import { ThemePicker } from "../theme";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { Badge } from "../components/ui/badge";

export async function loader({ request }: LoaderFunctionArgs) {
  if (!hasToken()) throw redirect("/connect");
  const options = { signal: request.signal, throwOnError: true as const };
  const identity = (await readIdentity(options)).data;
  const [projects, agents] = await Promise.all([
    listProjects(options),
    listAgents(options),
  ]);
  const url = new URL(request.url);
  const project =
    projects.data.find((p) => p.id === url.searchParams.get("project")) ??
    projects.data[0];
  const tasks = project
    ? (await listTasks({ ...options, path: { project_id: project.id! } })).data
    : [];
  const selected = tasks.find((t) => t.id === url.searchParams.get("task"));
  const detail = selected
    ? (await readTask({ ...options, path: { task_id: selected.id! } })).data
    : null;
  const system = url.searchParams.get("view") === "system";
  const warehouseView = url.searchParams.get("view") === "warehouse";
  const components = system
    ? (
        await listComponents({
          ...options,
          query: { environment: identity.environment },
        })
      ).data
    : [];
  return {
    identity,
    projects: projects.data,
    project,
    tasks,
    detail,
    agents: agents.data,
    components,
    system,
    warehouseView,
    exports: warehouseView ? (await listExports(options)).data : [],
    warehouse: warehouseView
      ? await readWarehouse(options)
          .then((r) => r.data)
          .catch(() => null)
      : null,
  };
}

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const value = (key: string) => String(form.get(key) ?? "");
  const intent = value("intent");
  const projectId = value("project");
  const taskId = value("task");
  const opts = { throwOnError: true as const, signal: request.signal };
  try {
    if (intent === "export") {
      const result = await createExport({
        ...opts,
        body: { request_key: value("request_key") },
      });
      return { exportId: result.data.id, error: undefined };
    }
    if (intent === "disconnect") {
      await disconnect();
      return redirect("/connect");
    }
    if (intent === "project") {
      const { data } = await createProject({
        ...opts,
        body: { name: value("name"), description: value("description") },
      });
      return redirect(`/?project=${data.id}`);
    }
    if (intent === "task") {
      const { data } = await createTask({
        ...opts,
        path: { project_id: projectId },
        body: {
          title: value("title"),
          prompt: value("prompt"),
          agent: value("agent"),
        },
      });
      return redirect(`/?project=${projectId}&task=${data.id}`);
    }
    if (intent === "message")
      await sendMessage({
        ...opts,
        path: { task_id: taskId },
        body: { content: value("content") },
      });
    if (intent === "cancel")
      await cancelTask({ ...opts, path: { task_id: taskId } });
    if (intent === "delete-task") {
      await deleteTask({ ...opts, path: { task_id: taskId } });
      return redirect(`/?project=${projectId}`);
    }
    if (intent === "delete-project") {
      await deleteProject({ ...opts, path: { project_id: projectId } });
      return redirect("/");
    }
    return { ok: true, error: "" };
  } catch (error) {
    const message =
      error &&
      typeof error === "object" &&
      "detail" in error &&
      typeof error.detail === "string"
        ? error.detail
        : "The request failed. Check your permissions and service status, then try again.";
    return { ok: false, error: message };
  }
}
const time = (value?: string) =>
  value
    ? new Date(value).toLocaleString([], {
        dateStyle: "short",
        timeStyle: "short",
      })
    : "—";
function Status({ value }: { value: string }) {
  return (
    <Badge variant="outline" className={`status status-${value}`}>
      {value.replaceAll("_", " ")}
    </Badge>
  );
}

export default function Console() {
  const exportData = useLoaderData<typeof loader>();
  const [exportKey, setExportKey] = useState(() => String(Date.now()));
  const {
    identity,
    projects,
    project,
    tasks,
    detail,
    agents,
    components,
    system,
    warehouseView,
  } = useLoaderData<typeof loader>();
  const mutation = useFetcher<typeof action>();
  const refresh = useRevalidator();
  const [newProject, setNewProject] = useState(false);
  const [newTask, setNewTask] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState("inspector");
  const taskAgent = agents.find((a) => a.id === detail?.task.agent);
  const draftAgent = agents.find((a) => a.id === selectedAgent);
  const [content, setContent] = useState("");
  const [artifact, setArtifact] = useState<{
    name: string;
    content: string;
  } | null>(null);
  const [artifactError, setArtifactError] = useState("");
  const writable = identity.scopes.includes("platform:verify");
  const busy = mutation.state !== "idle";
  const live =
    exportData.exports.some(
      (e) => e.status === "queued" || e.status === "running",
    ) ||
    tasks.some((t) => t.status === "queued" || t.status === "running") ||
    detail?.runs.some((r) => r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (mutation.data?.ok && !busy) setContent("");
  }, [mutation.data, busy]);
  useEffect(() => {
    setArtifact(null);
    setArtifactError("");
    setNewTask(false);
    setContent("");
  }, [detail?.task.id, project?.id]);
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => {
      if (!document.hidden && refresh.state === "idle")
        void refresh.revalidate();
    }, 2000);
    return () => clearInterval(timer);
  }, [live, refresh]);
  const fields = (
    <>
      <input type="hidden" name="project" value={project?.id ?? ""} />
      <input type="hidden" name="task" value={detail?.task.id ?? ""} />
    </>
  );
  return (
    <div className="workspace-shell">
      <aside className="sidebar">
        <Link to="/" className="wordmark">
          <Terminal size={18} /> platform
          <span className="text-muted-foreground">/ lab</span>
        </Link>
        <div className="sidebar-label">
          WORKSPACE <span>{identity.environment}</span>
        </div>
        <div className="sidebar-label">
          PROJECTS{" "}
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="New project"
            disabled={!writable}
            onClick={() => setNewProject(!newProject)}
          >
            <Plus />
          </Button>
        </div>
        {newProject && (
          <mutation.Form
            method="post"
            className="compact-form"
            onSubmit={() => setNewTask(false)}
          >
            <input type="hidden" name="intent" value="project" />
            <Input
              name="name"
              aria-label="Project name"
              placeholder="Project name"
              maxLength={120}
              required
            />
            <Textarea
              name="description"
              aria-label="Project description"
              placeholder="What are you exploring?"
              maxLength={2000}
            />
            <Button disabled={busy}>Create project</Button>
          </mutation.Form>
        )}
        <nav aria-label="Projects">
          {projects.map((p) => (
            <Link
              key={p.id}
              className={
                !system && !warehouseView && p.id === project?.id
                  ? "project-link selected"
                  : "project-link"
              }
              to={`/?project=${p.id}`}
            >
              <Folder size={14} />
              <span>{p.name}</span>
            </Link>
          ))}
        </nav>
        {!projects.length && (
          <p className="sidebar-empty">Create your first project to begin.</p>
        )}
        <div className="sidebar-bottom">
          <Link
            className={warehouseView ? "project-link selected" : "project-link"}
            to="/?view=warehouse"
          >
            <Database size={14} /> Warehouse
          </Link>
          <Link
            className={system ? "project-link selected" : "project-link"}
            to="/?view=system"
          >
            <Activity size={14} /> System
          </Link>
          <ThemePicker />
          <Form method="post">
            <Button
              variant="ghost"
              name="intent"
              value="disconnect"
              className="w-full justify-start"
            >
              Sign out
            </Button>
          </Form>
          <div className="identity">{identity.subject}</div>
        </div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-header">
          <span>
            Workspace <span className="slash">/</span>{" "}
            {warehouseView
              ? "Warehouse"
              : system
                ? "System"
                : (project?.name ?? "Projects")}
            {detail && !system && !warehouseView && (
              <>
                {" "}
                <span className="slash">/</span> Task
              </>
            )}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void refresh.revalidate()}
            disabled={refresh.state !== "idle"}
          >
            <RefreshCw size={14} /> Refresh
          </Button>
        </header>
        {mutation.data?.error && (
          <p className="error-banner" role="alert">
            {mutation.data.error}
          </p>
        )}
        {warehouseView ? (
          <section className="page">
            <div className="eyebrow">METADATA ANALYTICS</div>
            <h1>Warehouse</h1>
            <div className="panel-title">WAREHOUSE SNAPSHOTS</div>
            <p>
              Nightly at 02:00 UTC. Full metadata snapshots; conversation bodies
              remain in object storage.
            </p>
            <div className="snapshot-actions">
              <mutation.Form
                method="post"
                onSubmit={() => setExportKey(String(Date.now()))}
              >
                <input type="hidden" name="request_key" value={exportKey} />
                <Button
                  name="intent"
                  value="export"
                  disabled={!writable || busy}
                >
                  Export now
                </Button>
              </mutation.Form>
              <p>
                {exportData.warehouse?.export
                  ? `Latest validated snapshot: ${time(exportData.warehouse.export.updated_at)}`
                  : "No validated snapshot available yet."}
              </p>
            </div>
            <div className="system-table">
              {Object.entries(exportData.warehouse?.counts ?? {}).map(
                ([entity, count]) => (
                  <div key={entity}>
                    <strong>{entity}</strong>
                    <span>{count} rows in ClickHouse</span>
                  </div>
                ),
              )}
              {exportData.exports.map((e) => (
                <div key={e.id} data-export-id={e.id}>
                  <strong>
                    {e.trigger} · {time(e.created_at)}
                  </strong>
                  <span>
                    {e.id}
                    <br />
                    {e.workflow_name}
                    <br />
                    {e.error ?? (e.manifest_key || "Waiting for export")}
                  </span>
                  <Status value={e.status ?? "queued"} />
                </div>
              ))}
            </div>
          </section>
        ) : system ? (
          <section className="page">
            <div className="eyebrow">PLATFORM EVIDENCE</div>
            <h1>System</h1>
            <p className="lede">
              Live probes and integration status. Task execution is the
              end-to-end test.
            </p>
            <div className="panel-title">
              AGENT MODELS · CURRENT CONFIGURATION
            </div>
            <div className="system-table">
              {agents.map((a) => (
                <div key={a.id}>
                  <strong>{a.name}</strong>
                  <span>
                    {a.model} ·{" "}
                    {a.id === "inspector"
                      ? "Deterministic test model. Real platform tools; no external LLM calls."
                      : a.available
                        ? "Uses the configured model provider."
                        : "No model provider configured; tasks unavailable."}
                  </span>
                  <Status value={a.available ? "available" : "unavailable"} />
                </div>
              ))}
            </div>
            <div className="panel-title">COMPONENTS</div>
            <div className="system-table">
              {components.map((c) => (
                <div key={c.id}>
                  <strong>{c.name}</strong>
                  <span>{c.detail}</span>
                  <Status value={c.status} />
                </div>
              ))}
            </div>
          </section>
        ) : !project ? (
          <section className="page welcome">
            <div className="eyebrow">AGENT EXPERIMENTS</div>
            <h1>A workspace for work in progress.</h1>
            <p className="lede">
              Create a project. Give an agent a task. Follow the conversation,
              inspect its output, and try the next idea.
            </p>
            <Button onClick={() => setNewProject(true)} disabled={!writable}>
              <Plus size={16} /> Create a project
            </Button>
            <div className="welcome-steps">
              <div>
                <span>01</span>
                <h3>Organize</h3>
                <p>Keep related experiments together in a project.</p>
              </div>
              <div>
                <span>02</span>
                <h3>Delegate</h3>
                <p>Run tasks independently with the agent you choose.</p>
              </div>
              <div>
                <span>03</span>
                <h3>Inspect</h3>
                <p>
                  Read the conversation, artifacts, execution events, and trace
                  IDs.
                </p>
              </div>
            </div>
          </section>
        ) : detail ? (
          <section className="task-workspace">
            <div className="task-heading">
              <Link className="back-link" to={`/?project=${project.id}`}>
                <ArrowLeft size={14} /> All tasks
              </Link>
              <div className="title-line">
                <h1>{detail.task.title}</h1>
                <Status value={detail.task.status ?? "queued"} />
              </div>
              <div className="task-meta">
                {taskAgent?.name ?? detail.task.agent} <span>·</span>{" "}
                {detail.task.id}
              </div>
              <p className="form-hint">
                {detail.task.agent === "inspector"
                  ? "Pydantic AI TestModel · deterministic inspection using real platform tools. No external LLM calls."
                  : `Current model configuration: ${taskAgent?.model ?? "Unknown"}. Historical runs do not record their model.`}
              </p>
            </div>
            <div className="task-columns">
              <div className="conversation-column">
                <div className="panel-title">
                  CONVERSATION <span>{detail.messages.length} messages</span>
                </div>
                <div className="messages" aria-live="polite">
                  {detail.messages.map((m) => (
                    <article key={m.id} className="message">
                      <div className="message-label">
                        <strong>{m.role === "user" ? "You" : "Agent"}</strong>
                        <time>{time(m.created_at)}</time>
                      </div>
                      <div className="message-body">{m.content}</div>
                    </article>
                  ))}
                  {detail.runs.some((r) => r.status === "running") && (
                    <div className="working">
                      <span className="working-mark" /> Agent is working.
                      Follow-up instructions will queue another run.
                    </div>
                  )}
                </div>
                <mutation.Form method="post" className="composer">
                  {fields}
                  <input type="hidden" name="intent" value="message" />
                  <label htmlFor="follow-up">Continue the task</label>
                  <Textarea
                    id="follow-up"
                    name="content"
                    placeholder="Add context or give the agent its next instruction…"
                    required
                    maxLength={32000}
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    disabled={!writable}
                  />
                  <div className="composer-actions">
                    <span>
                      Messages and reports are stored in object storage.
                    </span>
                    <Button disabled={busy || !writable || !content.trim()}>
                      Send instruction <ArrowUpRight size={14} />
                    </Button>
                  </div>
                </mutation.Form>
              </div>
              <aside className="run-panel">
                <div className="panel-title">EXECUTION</div>
                <div className="run-actions">
                  <mutation.Form method="post">
                    {fields}
                    <Button
                      variant="outline"
                      size="sm"
                      name="intent"
                      value="cancel"
                      disabled={
                        busy ||
                        !writable ||
                        !detail.runs.some((r) =>
                          ["queued", "running"].includes(r.status ?? ""),
                        )
                      }
                    >
                      <Square size={12} /> Cancel runs
                    </Button>
                  </mutation.Form>
                  <mutation.Form
                    method="post"
                    onSubmit={(e) => {
                      if (
                        !confirm(
                          "Archive this task and cancel its active runs?",
                        )
                      )
                        e.preventDefault();
                    }}
                  >
                    {fields}
                    <Button
                      aria-label="Archive task"
                      variant="ghost"
                      size="icon-sm"
                      name="intent"
                      value="delete-task"
                      disabled={!writable || busy}
                    >
                      <Trash2 size={14} />
                    </Button>
                  </mutation.Form>
                </div>
                {detail.runs.map((r) => (
                  <div className="run" key={r.id}>
                    <div className="run-title">
                      <Status value={r.status ?? "queued"} />
                      <time>{time(r.created_at)}</time>
                    </div>
                    <code>{r.id}</code>
                    {r.id && detail.live_progress?.[r.id] && (
                      <div className="run-reference">
                        Live progress <code>{detail.live_progress[r.id]}</code>
                      </div>
                    )}
                    <div className="run-reference">
                      Model{" "}
                      <code>
                        {detail.task.agent === "inspector"
                          ? "TestModel"
                          : "Not recorded"}
                      </code>
                    </div>
                    {r.workflow_name && (
                      <div className="run-reference">
                        Workflow <code>{r.workflow_name}</code>
                      </div>
                    )}
                    {r.trace_id && (
                      <div className="run-reference">
                        Trace <code>{r.trace_id}</code>
                      </div>
                    )}
                    <span className="token-count">
                      {r.input_tokens ?? 0} input / {r.output_tokens ?? 0}{" "}
                      output tokens
                      {detail.task.agent === "inspector" &&
                        " · synthetic test usage, no LLM billing"}
                    </span>
                    {r.error && (
                      <p role="alert" className="text-destructive">
                        {r.error}
                      </p>
                    )}
                  </div>
                ))}
                <div className="panel-title">ARTIFACTS</div>
                {!detail.artifacts.length && (
                  <p className="panel-empty">
                    Reports appear here after a run completes.
                  </p>
                )}
                {detail.artifacts.map((a) => (
                  <Button
                    key={a.id}
                    variant="ghost"
                    className="artifact-link"
                    onClick={async () => {
                      try {
                        setArtifactError("");
                        const result = await readArtifact({
                          path: {
                            task_id: detail.task.id!,
                            artifact_id: a.id!,
                          },
                          throwOnError: true,
                        });
                        setArtifact({ name: a.name, content: result.data });
                      } catch {
                        setArtifactError("Could not retrieve the artifact.");
                      }
                    }}
                  >
                    {a.name}
                    <ArrowUpRight size={14} />
                  </Button>
                ))}
                {artifactError && <p role="alert">{artifactError}</p>}
                <div className="panel-title">EVENTS</div>
                {detail.events.map((e) => (
                  <div className="event" key={e.id}>
                    <span>{e.kind.replaceAll("_", " ")}</span>
                    <time>{time(e.created_at)}</time>
                  </div>
                ))}
              </aside>
            </div>
            {artifact && (
              <section className="artifact-preview">
                <div className="panel-title">
                  {artifact.name}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setArtifact(null)}
                  >
                    Close
                  </Button>
                </div>
                <pre>{artifact.content}</pre>
              </section>
            )}
          </section>
        ) : (
          <section className="page">
            <div className="eyebrow">PROJECT</div>
            <div className="title-line">
              <h1>{project.name}</h1>
              <div className="flex gap-2">
                <mutation.Form
                  method="post"
                  onSubmit={(e) => {
                    if (
                      !confirm(
                        "Archive this project and cancel all active tasks?",
                      )
                    )
                      e.preventDefault();
                  }}
                >
                  {fields}
                  <Button
                    name="intent"
                    value="delete-project"
                    variant="ghost"
                    size="icon"
                    aria-label="Archive project"
                    disabled={!writable || busy}
                  >
                    <Trash2 size={15} />
                  </Button>
                </mutation.Form>
                <Button
                  onClick={() => setNewTask(!newTask)}
                  disabled={!writable}
                >
                  <Plus size={15} /> New task
                </Button>
              </div>
            </div>
            <p className="lede">
              {project.description ||
                "Run experiments, compare agents, and build on what you learn."}
            </p>
            {newTask && (
              <mutation.Form method="post" className="new-task-form">
                {fields}
                <input type="hidden" name="intent" value="task" />
                <h2>Give an agent a task</h2>
                <label htmlFor="task-title">Task title</label>
                <Input
                  id="task-title"
                  name="title"
                  placeholder="What do you want to explore?"
                  required
                  maxLength={160}
                />
                <label htmlFor="agent">Agent</label>
                <select
                  id="agent"
                  name="agent"
                  value={selectedAgent}
                  onChange={(e) => setSelectedAgent(e.target.value)}
                  aria-describedby="agent-model-hint"
                >
                  {agents.map((a) => (
                    <option value={a.id} key={a.id} disabled={!a.available}>
                      {a.name} · {a.model}
                    </option>
                  ))}
                </select>
                <p className="form-hint" id="agent-model-hint">
                  {selectedAgent === "inspector"
                    ? "Pydantic AI TestModel runs real platform tools deterministically. No external LLM calls; token counts are synthetic."
                    : `Model: ${draftAgent?.model ?? "Not configured"}. Instructions and conversation history are sent to the configured model provider.`}
                </p>
                <label htmlFor="task-prompt">Instructions</label>
                <Textarea
                  id="task-prompt"
                  name="prompt"
                  rows={5}
                  placeholder="Describe the objective, useful context, and the output you expect."
                  required
                  maxLength={32000}
                />
                <div className="flex justify-end gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setNewTask(false)}
                  >
                    Cancel
                  </Button>
                  <Button disabled={busy}>
                    Create and run <ArrowUpRight size={14} />
                  </Button>
                </div>
              </mutation.Form>
            )}
            <div className="panel-title">
              TASKS <span>{tasks.length} total</span>
            </div>
            {!tasks.length ? (
              <div className="empty-state">
                <h2>No tasks yet.</h2>
                <p>
                  Start with an inspection of the platform, or configure a model
                  to run a research task.
                </p>
                <Button
                  variant="outline"
                  disabled={!writable}
                  onClick={() => setNewTask(true)}
                >
                  Create the first task
                </Button>
              </div>
            ) : (
              <div className="task-table">
                {tasks.map((t) => (
                  <Link
                    key={t.id}
                    to={`/?project=${project.id}&task=${t.id}`}
                    className="task-row"
                  >
                    <div>
                      <strong>{t.title}</strong>
                      <span>{agents.find((a) => a.id === t.agent)?.name}</span>
                    </div>
                    <Status value={t.status ?? "queued"} />
                    <time>{time(t.updated_at)}</time>
                    <ArrowUpRight size={14} />
                  </Link>
                ))}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

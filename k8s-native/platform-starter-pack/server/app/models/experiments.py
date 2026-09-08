from sqlmodel import Field
from ulid import ULID

from app.kit.db.models.base import RecordModel


class Project(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"prj_{ULID()}", primary_key=True)
    environment: str = Field(index=True)
    owner: str = Field(index=True)
    name: str
    description: str = ""


class Task(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"task_{ULID()}", primary_key=True)
    project_id: str = Field(foreign_key="project.id", index=True)
    title: str
    agent: str
    status: str = "queued"
    history_key: str | None = None


class Run(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"run_{ULID()}", primary_key=True)
    task_id: str = Field(foreign_key="task.id", index=True)
    prompt_key: str
    status: str = Field(default="queued", index=True)
    error: str | None = None
    workflow_name: str | None = None
    trace_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Message(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"msg_{ULID()}", primary_key=True)
    task_id: str = Field(foreign_key="task.id", index=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    role: str
    object_key: str
    content_type: str = "text/plain"
    size: int
    sha256: str


class Event(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"evt_{ULID()}", primary_key=True)
    task_id: str = Field(foreign_key="task.id", index=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    kind: str
    object_key: str | None = None


class Artifact(RecordModel, table=True):
    id: str = Field(default_factory=lambda: f"art_{ULID()}", primary_key=True)
    task_id: str = Field(foreign_key="task.id", index=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    name: str
    object_key: str
    content_type: str = "text/plain"
    size: int
    sha256: str

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models import Artifact, Event, Run, Task


class TaskCreate(SQLModel):
    title: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1, max_length=32000)
    agent: str = "inspector"


class MessageCreate(SQLModel):
    content: str = Field(min_length=1, max_length=32000)


class MessageRead(SQLModel):
    id: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    task_id: str
    run_id: str
    role: str
    content: str


class TaskDetail(SQLModel):
    task: Task
    messages: list[MessageRead]
    runs: list[Run]
    events: list[Event]
    artifacts: list[Artifact]
    live_progress: dict[str, str] = Field(default_factory=dict)


class AgentRead(SQLModel):
    id: str
    name: str
    description: str
    model: str
    available: bool

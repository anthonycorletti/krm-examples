from sqlmodel import Field, SQLModel


class ProjectCreate(SQLModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)

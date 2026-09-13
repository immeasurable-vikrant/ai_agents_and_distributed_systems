"""
models.py — the database layer AND the DTO layer.

Two distinct ideas live here, deliberately side by side so you can see
the difference:

  ORM MODELS (Task, Project)  → the shape data is STORED in.
                                Normalized. Optimized for writes + correctness.

  DTOs (TaskCreate, TaskOut)  → the shape data is TRANSFERRED in.
                                Flattened. Optimized for the consumer.

They are not the same shape, and that gap is the whole point.
"""

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import ForeignKey, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload

# ============================================================
# ENGINE — async, because every DB call is I/O we want to overlap
# ============================================================
# SQLite keeps this project to zero setup. The `+aiosqlite` driver is the
# ASYNC one — the sync driver would block the event loop on every query,
# freezing all concurrent requests. (Project 4 makes you measure that.)
# Swapping to Postgres is one line: postgresql+asyncpg://...
DATABASE_URL = "sqlite+aiosqlite:///./taskflow.db"

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ============================================================
# NORMALIZATION — why two tables instead of one
# ============================================================
#
# ❌ THE UNNORMALIZED VERSION (one table):
#
#   tasks(id, title, done, project_name, project_color)
#   ┌────┬──────────────┬──────────────┬───────────────┐
#   │ 1  │ Fix login    │ Work         │ blue          │
#   │ 2  │ Write tests  │ Work         │ blue          │   ← duplicated
#   │ 3  │ Buy milk     │ Personal     │ green         │
#   └────┴──────────────┴──────────────┴───────────────┘
#
#   Three concrete bugs this causes:
#
#   UPDATE anomaly — rename "Work" to "Office" and you must edit EVERY
#     row mentioning it. Miss one and your data contradicts itself.
#   INSERT anomaly — you cannot create an empty project. There's no row
#     to attach it to without inventing a fake task.
#   DELETE anomaly — delete the last "Personal" task and the project's
#     colour is gone from your system entirely.
#
# ✅ 3NF: project_name and project_color describe the PROJECT, not the
#    task. They depend on project_id — a non-key column — which is a
#    TRANSITIVE dependency. Move them to their own table.
#
#    "Every non-key column depends on the key, the whole key,
#     and nothing but the key."


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    color: Mapped[str] = mapped_column(String(20), default="grey")

    tasks: Mapped[list["Task"]] = relationship(back_populates="project")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    done: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # ONE-TO-MANY: the foreign key lives on the "many" side.
    # A project has many tasks; a task has one project.
    # index=True because we filter on it constantly — without an index
    # that's a full table scan every time.
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    project: Mapped["Project"] = relationship(back_populates="tasks")


# ============================================================
# DTOs — the API's shape, not the database's
# ============================================================
# Five reasons these exist separately from the models above:
#   1. SECURITY   — a model column you add later doesn't silently leak.
#   2. DECOUPLING — rename a DB column without breaking every client.
#   3. SHAPE      — the client wants project_name flat; the DB stores it
#                   normalized in another table. The DTO joins that gap.
#   4. VALIDATION — bad input is rejected before your code runs.
#   5. CONTRACT   — FastAPI generates /docs from these.

class TaskCreate(BaseModel):
    """What the client SENDS. No id, no created_at — the server owns those."""
    title: str = Field(..., min_length=1, max_length=200)
    project_id: int


class TaskUpdate(BaseModel):
    """
    PATCH = partial update, so every field is optional.

    NOTE it does NOT inherit TaskCreate. Inherited fields stay REQUIRED,
    so a PATCH sending only `done` would fail validation on `title`.
    Common bug; easy to avoid once you've seen it.
    """
    title: str | None = Field(None, min_length=1, max_length=200)
    done: bool | None = None


class TaskOut(BaseModel):
    """
    What the client RECEIVES — and here's the denormalization.

    The DB stores project_name in a separate table (3NF, no duplication).
    The client gets it flattened onto the task, because a UI rendering a
    task list shouldn't have to make a second request or do a join.

      NORMALIZED WRITES ─────► DENORMALIZED READS
           (models)              (this DTO)
    """
    id: int
    title: str
    done: bool
    created_at: datetime
    project_id: int
    project_name: str        # ← from the JOINed Project row
    project_color: str       # ← from the JOINed Project row

    model_config = {"from_attributes": True}

    @classmethod
    def build(cls, t: Task) -> "TaskOut":
        """
        Explicit mapper. Used instead of plain from_attributes because we
        need to REACH THROUGH the relationship to flatten fields.

        ⚠️ This requires t.project to be already loaded. If it isn't,
        async SQLAlchemy raises MissingGreenlet rather than lazy-loading.
        That's a feature: it turns a silent N+1 into a loud error.
        See list_tasks() below for the fix.
        """
        return cls(
            id=t.id, title=t.title, done=t.done, created_at=t.created_at,
            project_id=t.project_id,
            project_name=t.project.name,
            project_color=t.project.color,
        )


class ProjectOut(BaseModel):
    id: int
    name: str
    color: str
    task_count: int = 0
    model_config = {"from_attributes": True}


# ============================================================
# QUERIES — shared by the API and the agent's tools
# ============================================================
# Both the HTTP endpoints and the LangGraph tools call these. One place
# for the query logic means the agent can never bypass a rule the API
# enforces, and vice versa.

async def list_tasks(db: AsyncSession, done: bool | None = None) -> list[Task]:
    """
    selectinload() is doing real work here.

    WITHOUT it: 1 query for tasks, then ONE MORE per task when TaskOut.build
    touches t.project. 100 tasks = 101 queries. That's the N+1 problem —
    and in async SQLAlchemy it doesn't just get slow, it CRASHES.

    WITH it: 2 queries total, regardless of how many tasks.

    THE RULE: your eager-loading must match your DTO's nesting.
    """
    q = select(Task).options(selectinload(Task.project)).order_by(Task.id.desc())
    if done is not None:
        q = q.where(Task.done == done)
    return list((await db.execute(q)).scalars().all())


async def get_task(db: AsyncSession, task_id: int) -> Task | None:
    q = select(Task).options(selectinload(Task.project)).where(Task.id == task_id)
    return (await db.execute(q)).scalar_one_or_none()


async def list_projects(db: AsyncSession) -> list[Project]:
    q = select(Project).options(selectinload(Project.tasks)).order_by(Project.id)
    return list((await db.execute(q)).scalars().all())


async def find_project_by_name(db: AsyncSession, name: str) -> Project | None:
    q = select(Project).where(func.lower(Project.name) == name.lower().strip())
    return (await db.execute(q)).scalar_one_or_none()


async def init_db():
    """
    create_all() is a DEV shortcut — it builds tables from the models as
    they are right now, with no history and no way back.

    Project 3 replaces this with Alembic migrations, which are versioned,
    reversible, and can change a table that already holds real data.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed two projects so the app is usable immediately
    async with SessionLocal() as db:
        if not (await db.execute(select(Project))).scalars().first():
            db.add_all([
                Project(name="Work", color="blue"),
                Project(name="Personal", color="green"),
            ])
            await db.commit()
"""
main.py — FastAPI app: REST endpoints + the agent endpoint.

TWO WAYS IN, ONE SYSTEM:
  REST  → structured calls from a form/frontend
  Agent → plain English, which the LLM turns into the same operations

Both go through the same query layer in models.py. The agent isn't a
side door with its own rules — it's another client.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import models
from agent import graph_ascii, run_agent
from models import SessionLocal, Task, TaskCreate, TaskOut, TaskUpdate, ProjectOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    await models.init_db()
    yield


app = FastAPI(title="Project 2 — TaskFlow", lifespan=lifespan)


# ============================================================
# DEPENDENCY INJECTION
# ============================================================
# Depends(get_db) gives every endpoint a fresh session and — critically —
# the `yield` guarantees it's closed afterwards, even if the endpoint
# raises. Without that, connections leak until the pool is exhausted and
# the whole service stalls.
async def get_db():
    async with SessionLocal() as session:
        yield session


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/graph")
async def show_graph():
    return {"diagram": graph_ascii()}


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/api/tasks", response_model=list[TaskOut])
async def get_tasks(done: bool | None = None, db: AsyncSession = Depends(get_db)):
    """
    response_model=list[TaskOut] is the contract. Even if Task grows ten
    new columns tomorrow, only TaskOut's fields leave this endpoint.
    """
    tasks = await models.list_tasks(db, done=done)
    return [TaskOut.build(t) for t in tasks]


@app.post("/api/tasks", response_model=TaskOut, status_code=status.HTTP_201_CREATED)
async def add_task(body: TaskCreate, db: AsyncSession = Depends(get_db)):
    """201 Created, not 200 — the status code is part of the contract."""
    task = Task(title=body.title, project_id=body.project_id)
    db.add(task)
    await db.commit()

    # Re-fetch with the relationship eagerly loaded. refresh() alone
    # wouldn't populate task.project, and TaskOut.build needs it.
    full = await models.get_task(db, task.id)
    return TaskOut.build(full)


@app.patch("/api/tasks/{task_id}", response_model=TaskOut)
async def update_task(task_id: int, body: TaskUpdate, db: AsyncSession = Depends(get_db)):
    """
    PATCH = partial update. exclude_unset=True is what makes that work:
    it gives only the fields the client ACTUALLY SENT, so an unsent field
    keeps its value instead of being overwritten with None.
    """
    task = await models.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    await db.commit()

    return TaskOut.build(await models.get_task(db, task_id))


@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    task = await models.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    await db.delete(task)
    await db.commit()


@app.get("/api/projects", response_model=list[ProjectOut])
async def get_projects(db: AsyncSession = Depends(get_db)):
    projects = await models.list_projects(db)
    return [
        ProjectOut(id=p.id, name=p.name, color=p.color, task_count=len(p.tasks))
        for p in projects
    ]


# ============================================================
# AGENT ENDPOINT
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    intent: str          # which branch the router chose
    steps: int
    trace: list[dict]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Notice this endpoint takes NO db session. The agent's tools open
    their own — because they run inside the graph, not inside this
    request. That separation is what lets the same tools run later from
    a background worker with no HTTP request at all.
    """
    import os
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "OPENAI_API_KEY is not set")
    try:
        return ChatResponse(**await run_agent(body.message))
    except Exception as e:
        print(f"[agent] failed: {e}")
        raise HTTPException(500, f"Agent failed: {str(e)[:200]}")
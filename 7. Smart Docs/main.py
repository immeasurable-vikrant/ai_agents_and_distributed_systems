"""
main.py — FastAPI: auth, docs, and the three RAG modes.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field

import store
from rag import graph_ascii, run_rag
from store import TenantLeak


@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.init_db()
    yield


app = FastAPI(title="Project 7 — Smart Docs", lifespan=lifespan)


# ============================================================
# AUTH DEPENDENCY
# ============================================================

class Caller:
    def __init__(self, user_id: int, org_id: int):
        self.user_id = user_id
        self.org_id = org_id


async def current_user(authorization: str = Header(None)) -> Caller:
    """
    Every protected endpoint depends on this. Note there's NO database
    lookup — the token itself carries user_id and org_id, verified by
    signature. Stateless, so it scales without a session store.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    claims = store.read_token(authorization[7:])
    if not claims:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return Caller(claims["user_id"], claims["org_id"])


# ============================================================
# DTOs
# ============================================================

class LoginReq(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class DocReq(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=20, max_length=5000)
    # NOTE: no org_id field. Deliberately. It comes from the token —
    # letting a client name its own org would defeat the whole boundary.


class AskReq(BaseModel):
    question: str = Field(..., min_length=5, max_length=300)
    mode: Literal["naive", "corrective", "selfrag"] = "corrective"


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "real_embeddings": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/graph/{mode}")
async def show_graph(mode: str):
    if mode not in ("naive", "corrective", "selfrag"):
        raise HTTPException(404, "Unknown mode")
    return {"diagram": graph_ascii(mode)}


@app.post("/api/login")
async def login(body: LoginReq):
    result = await store.login(body.email, body.password)
    if not result:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return result


@app.get("/api/docs")
async def get_docs(me: Caller = Depends(current_user)):
    """Only this caller's org. Same filter as retrieval."""
    return await store.list_docs(me.org_id)


@app.post("/api/docs", status_code=status.HTTP_201_CREATED)
async def create_doc(body: DocReq, me: Caller = Depends(current_user)):
    doc_id = await store.add_doc(me.org_id, body.title, body.content)
    return {"id": doc_id, "org_id": me.org_id}


@app.post("/api/ask")
async def ask(body: AskReq, me: Caller = Depends(current_user)):
    """
    The org_id passed to the RAG graph comes from the VERIFIED TOKEN.
    There is no code path where a client supplies it.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "OPENAI_API_KEY required for the RAG graphs")
    try:
        return await run_rag(body.mode, body.question, me.org_id)
    except TenantLeak as e:
        # Layer 2 fired. This should be IMPOSSIBLE in normal operation —
        # if you see this in logs, a retrieval query's org filter is broken.
        # Return nothing useful; the incident detail goes to logs only.
        print(f"🚨 TENANT LEAK BLOCKED: {e}")
        raise HTTPException(500, "Internal error — this has been logged.")
    except Exception as e:
        print(f"[rag] {e}")
        raise HTTPException(500, f"RAG failed: {str(e)[:200]}")
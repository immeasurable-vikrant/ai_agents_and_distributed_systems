"""
auth_service.py — a standalone AUTH microservice.

Runs on :8001, owns its OWN database, and signs tokens with RS256.

WHY RS256 HERE AND NOT HS256 (Project 7's choice):

    HS256 (symmetric)              RS256 (asymmetric)
    ────────────────────────       ──────────────────────────────
    ONE shared secret              private key signs, public verifies
    every service that verifies    services verify with the PUBLIC key
    can also FORGE tokens          and CANNOT forge tokens
    fine for a monolith            required once services multiply

That's the whole argument. In a monolith, "the thing that verifies" and
"the thing that signs" are the same process, so a shared secret is fine.
The moment you have five services verifying tokens, a shared secret means
five services that could mint an admin token — and five places to leak it.

With RS256 only THIS service holds the private key. Everyone else gets
the public key, which is safe to publish (that's why /public-key needs
no auth).
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI, Header, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import ForeignKey, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ⭐ ITS OWN DATABASE. Not a schema in a shared DB — a separate file
# (separate Postgres instance in production). The RAG service literally
# cannot query these tables, which is the point.
DATABASE_URL = os.getenv("AUTH_DB_URL", "sqlite+aiosqlite:///./auth.db")
KEY_PATH = os.getenv("KEY_PATH", "./jwt_key.pem")

engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30), default="member")
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"), index=True)


# ============================================================
# KEYS — persisted, not regenerated on boot
# ============================================================

def load_or_create_keys():
    """
    ⚠️ THE KEY MUST PERSIST ACROSS RESTARTS.

    Generating a fresh keypair on every boot means every existing token
    becomes invalid the moment you redeploy — every user logged out, for
    no reason they can understand. Exercise 4 makes you delete this file
    and watch exactly that happen.

    In production this comes from a secrets manager, not a file on disk.
    """
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            private = serialization.load_pem_private_key(f.read(), password=None)
    else:
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(KEY_PATH, "wb") as f:
            f.write(private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ))

    priv_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


PRIVATE_KEY, PUBLIC_KEY = load_or_create_keys()


def make_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "org_id": user.org_id,
        "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iss": "auth-service",
    }
    return jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    async with SessionLocal() as db:
        if not (await db.execute(select(Org))).scalars().first():
            acme, globex = Org(name="Acme"), Org(name="Globex")
            db.add_all([acme, globex])
            await db.flush()
            db.add_all([
                User(email="alice@acme.com", org_id=acme.id, role="admin",
                     password_hash=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()),
                User(email="bob@globex.com", org_id=globex.id, role="member",
                     password_hash=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()),
            ])
            await db.commit()
    yield


app = FastAPI(title="Auth Service", lifespan=lifespan)


class LoginReq(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


@app.get("/health")
async def health():
    return {"service": "auth", "status": "ok"}


@app.post("/login")
async def login(body: LoginReq):
    async with SessionLocal() as db:
        u = (await db.execute(select(User).where(User.email == body.email))).scalar_one_or_none()
    if not u or not bcrypt.checkpw(body.password.encode()[:72], u.password_hash.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return {"token": make_token(u), "email": u.email,
            "org_id": u.org_id, "role": u.role}


@app.get("/public-key")
async def public_key():
    """
    Deliberately UNAUTHENTICATED. A public key is public — that's the
    entire premise of asymmetric crypto. Other services fetch this once,
    cache it, and verify tokens locally forever after.
    """
    return {"public_key": PUBLIC_KEY, "algorithm": "RS256"}


@app.post("/verify")
async def verify(authorization: str = Header(None)):
    """
    REMOTE verification — the OTHER strategy.

    Services can either:
      (a) fetch /public-key once and verify LOCALLY  — fast, no coupling
      (b) call THIS endpoint on every request        — slow, but central

    Both are implemented in rag_service.py so you can compare them
    directly. This one's advantage is instant revocation: if you delete a
    user, the very next request fails. With local verification, their
    token stays valid until it expires.

    Its cost is severe: every request in your whole system now depends on
    this service being up. Exercise 3 kills auth and shows what breaks
    under each strategy.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        claims = jwt.decode(authorization[7:], PUBLIC_KEY, algorithms=["RS256"])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")

    # Only remote verification can do this — check the user still exists.
    async with SessionLocal() as db:
        u = (await db.execute(select(User).where(User.id == int(claims["sub"])))).scalar_one_or_none()
    if not u:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")

    return {"valid": True, "user_id": u.id, "org_id": u.org_id, "role": u.role}
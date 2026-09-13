"""
migrations.py — versioned, reversible schema changes.

WHY THIS FILE EXISTS (and why Project 2's create_all() had to go):

    create_all()                    MIGRATIONS
    ────────────────────────        ──────────────────────────────
    "make the DB match models.py"   "apply these steps, in order"
    No history                      Full history, numbered
    No way back                     Every step has a downgrade()
    Only works on an EMPTY db       Works on a db with real rows
    Two devs = silent drift         Everyone runs the same steps

This is a MINIMAL migration runner — real projects use Alembic, and the
commands are shown at the bottom. The mechanics are identical: an ordered
list of (upgrade, downgrade) pairs, plus a table recording which have run.
Seeing it in 80 lines is more useful than seeing Alembic's 6-file scaffold.

RUN:
    python migrations.py upgrade      # apply everything pending
    python migrations.py status       # what's applied, what's not
    python migrations.py downgrade    # undo the last one
"""

import asyncio
import sys

from sqlalchemy import text

from store import engine


# ============================================================
# EXPAND → MIGRATE → CONTRACT
# ============================================================
# The goal: add a REQUIRED `summary` column to a table that already has
# rows. Doing it in one step fails — existing rows have no value for it.
#
# So you split it into three deployable steps:
#
#   0002 EXPAND    add the column as NULLABLE      → always safe, instant
#   0003 MIGRATE   backfill existing rows          → a data operation
#        CONTRACT  enforce NOT NULL                → safe now, every row has a value
#
# In production these ship as separate releases with time in between, so
# old and new application code can both run against the intermediate
# schema. That's what makes zero-downtime deploys possible.

MIGRATIONS = [
    (
        "0001_create_conversations",
        # upgrade
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   VARCHAR(64) NOT NULL UNIQUE,
            title       VARCHAR(200) DEFAULT 'New chat',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS ix_conv_thread ON conversations(thread_id);
        """,
        # downgrade
        "DROP TABLE IF EXISTS conversations;",
    ),
    (
        "0002_add_summary_nullable",
        # EXPAND — nullable, so it cannot fail on existing rows.
        # A NOT NULL column with no default would be rejected outright here.
        "ALTER TABLE conversations ADD COLUMN summary TEXT;",
        # SQLite can't DROP COLUMN in older versions; a real Postgres
        # downgrade would be: ALTER TABLE conversations DROP COLUMN summary;
        "-- (sqlite: drop column not supported in older versions)",
    ),
    (
        "0003_backfill_and_require_summary",
        # MIGRATE — give every existing row a value.
        # ⚠️ At real scale you'd batch this (WHERE id BETWEEN x AND y, in a
        # loop) — one unbounded UPDATE can lock a huge table for minutes.
        """
        UPDATE conversations SET summary = '' WHERE summary IS NULL;
        """,
        # CONTRACT would follow here. On Postgres:
        #     ALTER TABLE conversations ALTER COLUMN summary SET NOT NULL;
        # Safe NOW, because 0003 guaranteed no NULLs remain.
        "UPDATE conversations SET summary = NULL WHERE summary = '';",
    ),
]


async def _applied(conn) -> set[str]:
    await conn.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        " name VARCHAR(100) PRIMARY KEY,"
        " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    ))
    rows = await conn.execute(text("SELECT name FROM schema_version"))
    return {r[0] for r in rows}


async def upgrade():
    """Apply every migration that hasn't run yet, in order."""
    async with engine.begin() as conn:
        done = await _applied(conn)
        for name, up, _down in MIGRATIONS:
            if name in done:
                print(f"  ✓ {name} (already applied)")
                continue
            for stmt in filter(None, (s.strip() for s in up.split(";"))):
                if not stmt.startswith("--"):
                    await conn.execute(text(stmt))
            await conn.execute(
                text("INSERT INTO schema_version (name) VALUES (:n)"), {"n": name}
            )
            print(f"  ▲ {name} APPLIED")
    print("Schema is up to date.")


async def downgrade():
    """Undo the most recently applied migration."""
    async with engine.begin() as conn:
        done = await _applied(conn)
        for name, _up, down in reversed(MIGRATIONS):
            if name not in done:
                continue
            for stmt in filter(None, (s.strip() for s in down.split(";"))):
                if not stmt.startswith("--"):
                    await conn.execute(text(stmt))
            await conn.execute(
                text("DELETE FROM schema_version WHERE name = :n"), {"n": name}
            )
            print(f"  ▼ {name} REVERTED")
            return
    print("Nothing to revert.")


async def status():
    async with engine.begin() as conn:
        done = await _applied(conn)
    print("Migration status:")
    for name, _u, _d in MIGRATIONS:
        print(f"  [{'x' if name in done else ' '}] {name}")


# ============================================================
# THE REAL THING — Alembic commands, for when you graduate
# ============================================================
#   alembic init alembic                        set up the scaffold
#   alembic revision --autogenerate -m "msg"    diff models vs db, write a migration
#   alembic upgrade head                        apply all pending
#   alembic downgrade -1                        undo the last one
#   alembic current                             which version is the db at?
#   alembic history                             the full chain
#
# ⚠️ ALWAYS READ an --autogenerate output before applying it. It can't
# tell a RENAME from a DROP + ADD — which would silently delete a column
# of real data.

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    asyncio.run({"upgrade": upgrade, "downgrade": downgrade, "status": status}[cmd]())
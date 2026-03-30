"""
agent/database.py
─────────────────
PostgreSQL async database — replaces SQLite.

Same API as before — zero logic changes anywhere else.
Uses asyncpg via databases library for async PostgreSQL.

Storage: DATABASE_URL environment variable (Render PostgreSQL)
Fallback: SQLite (./etgenai.db) if DATABASE_URL not set — local dev works too!

Tables:
  chats      — every user/ai message per thread
  summaries  — rolling conversation summary per thread

Install:
  pip install databases[asyncpg] asyncpg
"""

import os
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
_SQLITE_PATH = os.getenv("SQLITE_PATH", "./etgenai.db")

# ── Detect which backend to use ───────────────────────────────────────────────
_USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))

if _USE_POSTGRES:
    import databases
    # Render gives postgresql:// but asyncpg needs postgresql+asyncpg://
    _DB_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    _db = databases.Database(_DB_URL)
    logger.info(f"[DB] Using PostgreSQL")
else:
    import aiosqlite
    logger.info(f"[DB] DATABASE_URL not set — using SQLite → {_SQLITE_PATH}")


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS chats (
    id         SERIAL PRIMARY KEY,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chats_thread ON chats(thread_id);

CREATE TABLE IF NOT EXISTS summaries (
    thread_id  TEXT PRIMARY KEY,
    summary    TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chats_thread ON chats(thread_id);
CREATE TABLE IF NOT EXISTS summaries (
    thread_id  TEXT PRIMARY KEY,
    summary    TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ── Init ──────────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create tables. Called once on app startup."""
    if _USE_POSTGRES:
        await _db.connect()
        for stmt in _DDL_POSTGRES.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await _db.execute(stmt)
        logger.info("[DB] PostgreSQL ready ✓")
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            await db.executescript(_DDL_SQLITE)
            await db.commit()
        logger.info(f"[DB] SQLite ready → {_SQLITE_PATH}")


# ── Save message ──────────────────────────────────────────────────────────────

async def save_message(thread_id: str, role: str, content: str) -> None:
    """Persist a single chat message."""
    if _USE_POSTGRES:
        await _db.execute(
            "INSERT INTO chats (thread_id, role, content) VALUES (:thread_id, :role, :content)",
            {"thread_id": thread_id, "role": role, "content": content},
        )
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            await db.execute(
                "INSERT INTO chats (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content),
            )
            await db.commit()


# ── Load messages ─────────────────────────────────────────────────────────────

async def load_messages(thread_id: str) -> list[dict]:
    """Return all messages for a thread in chronological order."""
    if _USE_POSTGRES:
        rows = await _db.fetch_all(
            "SELECT role, content, created_at::text FROM chats "
            "WHERE thread_id = :thread_id ORDER BY id ASC",
            {"thread_id": thread_id},
        )
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                for r in rows]
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT role, content, created_at FROM chats "
                "WHERE thread_id = ? ORDER BY id ASC",
                (thread_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                for r in rows]


# ── Save summary ──────────────────────────────────────────────────────────────

async def save_summary(thread_id: str, summary: str) -> None:
    """Upsert rolling conversation summary."""
    if _USE_POSTGRES:
        await _db.execute(
            """
            INSERT INTO summaries (thread_id, summary, updated_at)
            VALUES (:thread_id, :summary, NOW())
            ON CONFLICT (thread_id) DO UPDATE
            SET summary = EXCLUDED.summary, updated_at = NOW()
            """,
            {"thread_id": thread_id, "summary": summary},
        )
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            await db.execute(
                """
                INSERT INTO summaries (thread_id, summary, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(thread_id) DO UPDATE
                SET summary = excluded.summary, updated_at = datetime('now')
                """,
                (thread_id, summary),
            )
            await db.commit()


# ── Load summary ──────────────────────────────────────────────────────────────

async def load_summary(thread_id: str) -> str:
    """Return stored summary or empty string."""
    if _USE_POSTGRES:
        row = await _db.fetch_one(
            "SELECT summary FROM summaries WHERE thread_id = :thread_id",
            {"thread_id": thread_id},
        )
        return row["summary"] if row else ""
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT summary FROM summaries WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row["summary"] if row else ""


# ── List threads ──────────────────────────────────────────────────────────────

async def list_threads() -> list[dict]:
    """Return all threads with message count and last activity."""
    if _USE_POSTGRES:
        rows = await _db.fetch_all(
            """
            SELECT thread_id, COUNT(*) AS message_count,
                   MAX(created_at)::text AS last_active
            FROM chats GROUP BY thread_id
            ORDER BY last_active DESC LIMIT 100
            """
        )
        return [{"thread_id": r["thread_id"], "message_count": r["message_count"],
                 "last_active": r["last_active"]} for r in rows]
    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT thread_id, COUNT(*) AS message_count,
                       MAX(created_at) AS last_active
                FROM chats GROUP BY thread_id
                ORDER BY last_active DESC LIMIT 100
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [{"thread_id": r["thread_id"], "message_count": r["message_count"],
                 "last_active": r["last_active"]} for r in rows]


# ── Debug thread ──────────────────────────────────────────────────────────────

async def debug_thread(thread_id: str) -> dict:
    """Diagnostic info for a thread."""
    if _USE_POSTGRES:
        count_row = await _db.fetch_one(
            "SELECT COUNT(*) AS cnt FROM chats WHERE thread_id = :thread_id",
            {"thread_id": thread_id},
        )
        count = count_row["cnt"] if count_row else 0

        summary_row = await _db.fetch_one(
            "SELECT summary FROM summaries WHERE thread_id = :thread_id",
            {"thread_id": thread_id},
        )

        last_rows = await _db.fetch_all(
            "SELECT role, content, created_at::text FROM chats "
            "WHERE thread_id = :thread_id ORDER BY id DESC LIMIT 5",
            {"thread_id": thread_id},
        )
        last_5 = [{"role": r["role"], "content": r["content"][:120],
                   "created_at": r["created_at"]} for r in last_rows]

    else:
        import aiosqlite
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT COUNT(*) AS cnt FROM chats WHERE thread_id = ?",
                (thread_id,),
            ) as cur:
                count = (await cur.fetchone())["cnt"]
            async with db.execute(
                "SELECT summary FROM summaries WHERE thread_id = ?",
                (thread_id,),
            ) as cur:
                summary_row = await cur.fetchone()
            async with db.execute(
                "SELECT role, content, created_at FROM chats "
                "WHERE thread_id = ? ORDER BY id DESC LIMIT 5",
                (thread_id,),
            ) as cur:
                last_5 = [{"role": r["role"], "content": r["content"][:120],
                           "created_at": r["created_at"]} for r in await cur.fetchall()]

    return {
        "thread_id":        thread_id,
        "db_backend":       "postgresql" if _USE_POSTGRES else "sqlite",
        "db_message_count": count,
        "summary_exists":   summary_row is not None,
        "last_5_messages":  last_5,
        "diagnosis": "Thread has messages" if count > 0 else "Thread not found",
    }


# ── Close ─────────────────────────────────────────────────────────────────────

async def close_db() -> None:
    """Close database connection."""
    if _USE_POSTGRES:
        await _db.disconnect()
        logger.info("[DB] PostgreSQL disconnected")
    else:
        logger.info("[DB] SQLite — no connection to close")
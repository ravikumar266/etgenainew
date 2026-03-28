"""
agent/database.py
─────────────────
Pure SQLite async database — zero PostgreSQL dependency.

Storage: ./etgenai.db  (created automatically on first run)
Override path via SQLITE_PATH in .env

Tables:
  chats      — every user/ai message per thread
  summaries  — rolling conversation summary per thread
"""

import os
import logging

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH: str = os.getenv("SQLITE_PATH", "./etgenai.db")


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chats_thread
    ON chats(thread_id);

CREATE TABLE IF NOT EXISTS summaries (
    thread_id  TEXT PRIMARY KEY,
    summary    TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ── Public API ────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create tables if they don't exist. Called once on app startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_DDL)
        await db.commit()
    logger.info(f"[DB] SQLite ready → {DB_PATH}")


async def save_message(thread_id: str, role: str, content: str) -> None:
    """Persist a single chat message."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chats (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        await db.commit()


async def load_messages(thread_id: str) -> list[dict]:
    """Return all messages for a thread in chronological order."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content, created_at FROM chats "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        for r in rows
    ]


async def save_summary(thread_id: str, summary: str) -> None:
    """Upsert the rolling conversation summary for a thread."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO summaries (thread_id, summary, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(thread_id) DO UPDATE
            SET summary = excluded.summary,
                updated_at = datetime('now')
            """,
            (thread_id, summary),
        )
        await db.commit()


async def load_summary(thread_id: str) -> str:
    """Return the stored summary for a thread, or empty string."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT summary FROM summaries WHERE thread_id = ?",
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return row["summary"] if row else ""


async def list_threads() -> list[dict]:
    """Return all threads with message count and last activity."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT thread_id,
                   COUNT(*) AS message_count,
                   MAX(created_at) AS last_active
            FROM chats
            GROUP BY thread_id
            ORDER BY last_active DESC
            LIMIT 100
            """
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {
            "thread_id":     r["thread_id"],
            "message_count": r["message_count"],
            "last_active":   r["last_active"],
        }
        for r in rows
    ]


async def debug_thread(thread_id: str) -> dict:
    """Diagnostic info for a thread — used by /debug/memory endpoint."""
    async with aiosqlite.connect(DB_PATH) as db:
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
            last_5 = [
                {
                    "role":       r["role"],
                    "content":    r["content"][:120],
                    "created_at": r["created_at"],
                }
                for r in await cur.fetchall()
            ]

    return {
        "thread_id":        thread_id,
        "db_backend":       "sqlite",
        "db_path":          DB_PATH,
        "db_message_count": count,
        "summary_exists":   summary_row is not None,
        "last_5_messages":  last_5,
        "diagnosis": (
            "Thread has messages" if count > 0
            else "Thread not found or no messages yet"
        ),
    }


async def close_db() -> None:
    """No-op for SQLite — connections are opened/closed per query."""
    logger.info("[DB] SQLite — no persistent connection to close")
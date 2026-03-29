"""
agent/memory_manager.py
────────────────────────
Sliding-window + summarization with in-memory cache.

Fix 6 applied:
  - Summary cached in memory dict — no SQLite read on every turn
  - Cache invalidated only when new summary is written
  - Eliminates repeated DB reads for active sessions
"""

import os
import logging
from typing import Optional

from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)

from agent.database import save_summary, load_summary

logger = logging.getLogger(__name__)

MAX_MESSAGES: int = int(os.getenv("MEMORY_MAX_MESSAGES", "20"))
KEEP_LAST_K:  int = int(os.getenv("MEMORY_KEEP_LAST_K",  "8"))
MAX_TOKENS:   int = int(os.getenv("MEMORY_MAX_TOKENS",   "3000"))
_CHARS_PER_TOKEN = 3.5

# ── In-memory summary cache (Fix 6) ──────────────────────────────────────────
# thread_id → summary text
# Populated on first load, updated on write, never hits SQLite repeatedly
_summary_cache: dict[str, str] = {}


def _estimate_tokens(text) -> int:
    text = str(text) if not isinstance(text, str) else text
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _msg_text(msg: AnyMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content).strip() if content is not None else ""


def _msg_label(msg: AnyMessage) -> str:
    if isinstance(msg, SystemMessage):  return "SYSTEM"
    if isinstance(msg, HumanMessage):   return "USER"
    if isinstance(msg, AIMessage):      return "ASSISTANT"
    if isinstance(msg, ToolMessage):    return f"TOOL({getattr(msg, 'name', '?')})"
    return msg.type.upper()


async def _get_summary_cached(thread_id: str) -> str:
    """Load summary from cache first, SQLite only on cache miss."""
    if thread_id in _summary_cache:
        return _summary_cache[thread_id]
    try:
        summary = await load_summary(thread_id)
        _summary_cache[thread_id] = summary
        return summary
    except Exception as e:
        logger.warning(f"[Memory] Could not load summary: {e}")
        return ""


async def _save_summary_cached(thread_id: str, summary: str) -> None:
    """Save to SQLite and update cache."""
    _summary_cache[thread_id] = summary
    try:
        await save_summary(thread_id, summary)
    except Exception as e:
        logger.warning(f"[Memory] Could not save summary: {e}")


async def maybe_summarize(
    thread_id: str,
    messages:  list[AnyMessage],
    llm,
) -> list[AnyMessage]:
    """Summarize + trim if over MAX_MESSAGES. Uses cached summary."""
    non_system  = [m for m in messages if not isinstance(m, SystemMessage)]
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]

    if len(non_system) <= MAX_MESSAGES:
        return messages

    cutoff       = len(non_system) - KEEP_LAST_K
    to_summarize = non_system[:cutoff]
    to_keep      = non_system[cutoff:]

    logger.info(
        f"[Memory] '{thread_id}': {len(non_system)} msgs → "
        f"summarizing {len(to_summarize)}, keeping {len(to_keep)}"
    )

    existing_summary = await _get_summary_cached(thread_id)

    try:
        conversation_block = ""
        for msg in to_summarize:
            text = _msg_text(msg)
            if text:
                max_chars = int(MAX_TOKENS * _CHARS_PER_TOKEN // max(len(to_summarize), 1))
                conversation_block += f"{_msg_label(msg)}: {text[:max_chars]}\n\n"

        prior = f"EXISTING SUMMARY:\n{existing_summary}\n\n" if existing_summary else ""
        prompt = (
            f"{prior}CONVERSATION TO SUMMARIZE:\n{conversation_block}"
            "Write a concise summary covering: topics discussed, decisions made, "
            "important facts, pending tasks, and user goals. Plain paragraphs only."
        )

        summary_msg = llm.invoke(prompt)
        new_summary = _msg_text(summary_msg) if hasattr(summary_msg, "content") else str(summary_msg)

        if not new_summary.strip():
            raise ValueError("Empty summary")

        await _save_summary_cached(thread_id, new_summary)
        logger.info(f"[Memory] Summary saved ({len(new_summary)} chars)")

    except Exception as e:
        logger.error(f"[Memory] Summarization failed: {e} — keeping all")
        return messages

    return system_msgs + to_keep


async def inject_summary_context(
    thread_id: str,
    messages:  list[AnyMessage],
) -> list[AnyMessage]:
    """Inject cached summary as context — no repeated SQLite reads."""
    summary = await _get_summary_cached(thread_id)

    if not summary.strip():
        return messages

    insert_at = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            insert_at = i + 1
        else:
            break

    if insert_at < len(messages):
        if _msg_text(messages[insert_at]).startswith("[CONVERSATION SUMMARY]"):
            return messages

    summary_msg = HumanMessage(content=(
        f"[CONVERSATION SUMMARY]\n{summary}\n[END SUMMARY]"
    ))
    ack_msg = AIMessage(content="Understood. Continuing from our earlier conversation.")

    result = messages[:insert_at] + [summary_msg, ack_msg] + messages[insert_at:]
    logger.info(f"[Memory] Summary injected ({len(summary)} chars) for '{thread_id}'")
    return result


def estimate_message_tokens(messages: list[AnyMessage]) -> int:
    total_chars = sum(len(_msg_text(m)) for m in messages if hasattr(m, "content"))
    return max(1, int(total_chars / _CHARS_PER_TOKEN))
"""
agent/memory_manager.py
────────────────────────
Sliding-window + summarization memory management.

Strategy:
  1. After every LLM response, count messages in the current state.
  2. If message count > MAX_MESSAGES, summarize the oldest
     (total - KEEP_LAST_K) messages using the LLM.
  3. Store the updated summary in SQLite (summaries table).
  4. Drop the summarized messages from the LangGraph state,
     keeping only the last KEEP_LAST_K messages + SystemMessage.
  5. On every chat_node entry, prepend a SummaryMessage (HumanMessage
     containing the stored summary) so the model has full context.

Configuration (via .env or environment variables):
  MEMORY_MAX_MESSAGES   — trigger summarization above this count  (default: 20)
  MEMORY_KEEP_LAST_K    — messages to keep after summarization    (default: 8)
  MEMORY_MAX_TOKENS     — soft token budget for summary prompt    (default: 3000)
"""

import os
import logging
from typing import Optional

from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)

from agent.database import save_summary, load_summary

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_MESSAGES: int = int(os.getenv("MEMORY_MAX_MESSAGES", "20"))
KEEP_LAST_K:  int = int(os.getenv("MEMORY_KEEP_LAST_K",  "8"))
MAX_TOKENS:   int = int(os.getenv("MEMORY_MAX_TOKENS",   "3000"))

# Rough chars-per-token estimate (conservative for mixed content)
_CHARS_PER_TOKEN = 3.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text) -> int:
    text = str(text) if not isinstance(text, str) else text
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _msg_text(msg: AnyMessage) -> str:
    """Extract plain text from any message type — always returns a string."""
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    # Fallback: stringify whatever came back
    return str(content).strip() if content is not None else ""


def _msg_label(msg: AnyMessage) -> str:
    """Human-readable role label for the summary prompt."""
    if isinstance(msg, SystemMessage):
        return "SYSTEM"
    if isinstance(msg, HumanMessage):
        return "USER"
    if isinstance(msg, AIMessage):
        return "ASSISTANT"
    if isinstance(msg, ToolMessage):
        return f"TOOL({getattr(msg, 'name', '?')})"
    return msg.type.upper()


def _build_summary_prompt(existing_summary: str, messages_to_summarize: list[AnyMessage]) -> str:
    """Build the prompt sent to the LLM to produce/update the summary."""
    conversation_block = ""
    for msg in messages_to_summarize:
        label = _msg_label(msg)
        text  = _msg_text(msg)
        if text:
            # Truncate individual messages so the prompt stays under MAX_TOKENS
            max_chars = int(MAX_TOKENS * _CHARS_PER_TOKEN // max(len(messages_to_summarize), 1))
            conversation_block += f"{label}: {text[:max_chars]}\n\n"

    prior_block = (
        f"EXISTING SUMMARY (from earlier in the conversation):\n{existing_summary}\n\n"
        if existing_summary else ""
    )

    return (
        f"{prior_block}"
        "CONVERSATION SEGMENT TO SUMMARIZE:\n"
        f"{conversation_block}"
        "──────────────────────────────────\n"
        "Produce a concise but complete summary of the conversation above. "
        "Include: key topics discussed, decisions made, important facts shared, "
        "any pending tasks or follow-ups, and the user's goals. "
        "Write in third person. Be specific — include names, numbers, and details "
        "that would be needed to continue this conversation accurately. "
        "Do NOT include filler phrases like 'The conversation covered…'. "
        "Just output the summary directly, in plain paragraphs."
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def maybe_summarize(
    thread_id: str,
    messages:  list[AnyMessage],
    llm,                          # any LangChain chat model
) -> list[AnyMessage]:
    """
    Check if the message list exceeds MAX_MESSAGES.
    If so, summarize the oldest messages, persist the summary,
    and return a trimmed message list.

    Always returns a valid message list (never raises).
    """
    # Filter out SystemMessages from the count — they're always kept separately
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]

    if len(non_system) <= MAX_MESSAGES:
        return messages  # Nothing to do

    # Split: messages to summarize vs messages to keep
    cutoff            = len(non_system) - KEEP_LAST_K
    to_summarize      = non_system[:cutoff]
    to_keep           = non_system[cutoff:]

    logger.info(
        f"[Memory] Thread '{thread_id}': {len(non_system)} messages → "
        f"summarizing {len(to_summarize)}, keeping last {len(to_keep)}"
    )

    # Load existing summary so we can extend it incrementally
    try:
        existing_summary = await load_summary(thread_id)
    except Exception as e:
        logger.warning(f"[Memory] Could not load existing summary: {e}")
        existing_summary = ""

    # Build summary prompt and call LLM
    try:
        prompt      = _build_summary_prompt(existing_summary, to_summarize)
        token_est   = _estimate_tokens(prompt)
        logger.info(f"[Memory] Summary prompt ≈ {token_est} tokens")

        summary_msg = llm.invoke(prompt)
        new_summary = _msg_text(summary_msg) if hasattr(summary_msg, "content") else str(summary_msg)

        if not new_summary.strip():
            raise ValueError("LLM returned empty summary")

        logger.info(f"[Memory] Summary generated ({len(new_summary)} chars)")

    except Exception as e:
        logger.error(f"[Memory] Summarization failed: {e} — keeping all messages")
        return messages  # Fail-safe: keep everything

    # Persist summary to SQLite
    try:
        await save_summary(thread_id, new_summary)
        logger.info(f"[Memory] Summary saved for thread '{thread_id}'")
    except Exception as e:
        logger.warning(f"[Memory] Could not save summary: {e}")

    # Reconstruct message list: system + kept messages
    # The summary is injected as context in inject_summary_context() below,
    # so we don't duplicate it here.
    trimmed = system_msgs + to_keep
    logger.info(
        f"[Memory] Trimmed state: {len(messages)} → {len(trimmed)} messages "
        f"({len(to_summarize)} summarized)"
    )
    return trimmed


async def inject_summary_context(
    thread_id: str,
    messages:  list[AnyMessage],
) -> list[AnyMessage]:
    """
    Load the stored summary for this thread and inject it as a
    HumanMessage/AIMessage pair right after the SystemMessage (if any).

    This ensures the LLM always has the summarized context without
    bloating the live message list.

    Returns the (possibly augmented) message list.
    """
    try:
        summary = await load_summary(thread_id)
    except Exception as e:
        logger.warning(f"[Memory] Could not load summary for injection: {e}")
        return messages

    if not summary.strip():
        return messages  # Nothing to inject

    # Find insertion point: right after SystemMessage(s)
    insert_at = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            insert_at = i + 1
        else:
            break

    # Avoid duplicate injection (check if summary marker already present)
    if insert_at < len(messages):
        next_msg_text = _msg_text(messages[insert_at])
        if next_msg_text.startswith("[CONVERSATION SUMMARY]"):
            return messages  # Already injected

    summary_injection = HumanMessage(
        content=(
            "[CONVERSATION SUMMARY — context from earlier in this conversation]\n"
            f"{summary}\n"
            "[END OF SUMMARY — continue the conversation from here]"
        )
    )
    # Add a brief AI acknowledgement so the turn structure stays valid
    ack_injection = AIMessage(
        content="Understood. I have the context from our earlier conversation and will continue accordingly."
    )

    result = messages[:insert_at] + [summary_injection, ack_injection] + messages[insert_at:]
    logger.info(f"[Memory] Injected summary context ({len(summary)} chars) for thread '{thread_id}'")
    return result


def estimate_message_tokens(messages: list[AnyMessage]) -> int:
    """Rough token estimate for the full message list."""
    total_chars = sum(len(_msg_text(m)) for m in messages if hasattr(m, "content"))
    return max(1, int(total_chars / _CHARS_PER_TOKEN))
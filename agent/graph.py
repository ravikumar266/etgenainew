"""
graph.py — LangGraph multi-agent orchestration

Checkpointer: MemorySaver (in-process SQLite via LangGraph)
  - No PostgreSQL needed
  - Conversation history persists within a running session
  - On restart, history is lost (use /history endpoint which reads from etgenai.db)

Memory Management (NEW):
  - Sliding-window + summarization via agent/memory_manager.py
  - When messages exceed MEMORY_MAX_MESSAGES (default 20), older messages
    are summarized by the LLM and stored in SQLite (summaries table).
  - Only the last MEMORY_KEEP_LAST_K (default 8) messages are kept in state.
  - The summary is re-injected as context on every chat_node entry.
  - Configure via .env: MEMORY_MAX_MESSAGES, MEMORY_KEEP_LAST_K, MEMORY_MAX_TOKENS

Agents:
  chat_node          — Primary agent: Gemini with all tools
  email_approval     — Human-in-the-loop gate for send_email
  tool_node          — Executes all tool calls
  critic_node        — Quality verification agent (second LLM pass)

Flow:
  chat_node → [tool calls?]
    → NO  → critic_node → END
    → YES → [is send_email?]
                → YES → email_approval_node → tools → chat_node → critic_node
                → NO  → tool_node → chat_node → critic_node
"""

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from langchain_core.runnables import RunnableConfig

from agent.config import llm, logger
from agent.prompts import SYSTEM_PROMPT
from agent.memory_manager import maybe_summarize, inject_summary_context, estimate_message_tokens

# ── Tool imports ──────────────────────────────────────────────────────────────
from agent.tools_search import search_web, fetch_webpage, duckduckgo_search
from agent.tools_google import get_weather, google_doc, update_google_doc
from agent.tools_email import send_email, check_updates
from agent.tools_github import review_pr, get_pr_files, list_prs, get_file, search_code
from agent.tools_rag import (
    ingest_webpage, ingest_pdf, ingest_youtube,
    query_rag, list_rag_collections, delete_rag_collection,
)
from agent.tools_workflow import (
    start_workflow, update_workflow_step,
    get_workflow_status, list_workflows, escalate_workflow,
)
from agent.tools_meeting import (
    process_meeting, check_action_items, escalate_stalled_items,
)
from agent.scheduler import start_scheduler, stop_scheduler, get_scheduler  # noqa: F401

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    search_web, fetch_webpage, duckduckgo_search,
    get_weather,
    google_doc, update_google_doc,
    send_email, check_updates,
    review_pr, get_pr_files, list_prs, get_file, search_code,
    ingest_webpage, ingest_pdf, ingest_youtube,
    query_rag, list_rag_collections, delete_rag_collection,
    start_workflow, update_workflow_step,
    get_workflow_status, list_workflows, escalate_workflow,
    process_meeting, check_action_items, escalate_stalled_items,
]

_llm_with_tools = llm.bind_tools(TOOLS)
_critic_llm     = llm


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages:       Annotated[list[AnyMessage], add_messages]
    critic_score:   int
    critic_retries: int
    thread_id:      Optional[str]   # NEW: carried through state for memory ops


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _get_thread_id(config: RunnableConfig) -> str:
    """Extract thread_id from LangGraph RunnableConfig."""
    try:
        cfg = config if isinstance(config, dict) else {}
        return cfg.get("configurable", {}).get("thread_id", "default")
    except Exception:
        return "default"


def _has_rag_notice(messages: list) -> bool:
    """Check if a RAG-ready notice is already in the message list."""
    for m in messages:
        if isinstance(m, HumanMessage):
            txt = _extract_text(m.content)
            if "[RAG CONTEXT AVAILABLE]" in txt:
                return True
    return False


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def chat_node(state: State, config: RunnableConfig) -> State:
    """
    Primary agent — Gemini with all tools.

    Memory flow:
      1. maybe_summarize() — if too many messages, summarize + trim state
      2. inject_summary_context() — prepend stored summary as context
      3. inject RAG notice if collections exist for this thread
      4. Ensure SystemMessage is first
      5. Invoke LLM
    """
    thread_id = _get_thread_id(config)
    messages  = list(state["messages"])

    # ── Step 1: Summarize + trim if over the limit ────────────────────────────
    token_est = estimate_message_tokens(messages)
    logger.info(f"[Chat] Thread '{thread_id}': {len(messages)} messages ≈ {token_est} tokens")

    messages = await maybe_summarize(thread_id, messages, llm)

    # ── Step 2: Inject stored summary context ─────────────────────────────────
    messages = await inject_summary_context(thread_id, messages)

    # ── Step 3: Ensure SystemMessage is always first ──────────────────────────
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    # ── Step 4: Inject RAG availability notice ────────────────────────────────
    # If user uploaded a PDF/webpage for this thread, remind the agent to use query_rag
    if not _has_rag_notice(messages):
        try:
            from agent.tools_rag import list_rag_collections
            result = list_rag_collections.invoke({})
            result_str = str(result)
            # Check if this thread's collection has data
            if thread_id in result_str or "default" in result_str:
                rag_notice = HumanMessage(content=(
                    f"[RAG CONTEXT AVAILABLE] Documents have been ingested into the RAG "
                    f"collection '{thread_id}'. When the user asks about any uploaded document "
                    f"or PDF, ALWAYS call the query_rag tool with collection='{thread_id}' "
                    f"to retrieve relevant content. Do NOT ask for a file path — the document "
                    f"is already indexed and ready to query."
                ))
                rag_ack = AIMessage(content=(
                    f"Understood. I have access to ingested documents in collection "
                    f"'{thread_id}' and will query them automatically when relevant."
                ))
                # Insert after system message
                messages = messages[:1] + [rag_notice, rag_ack] + messages[1:]
                logger.info(f"[Chat] RAG notice injected for collection '{thread_id}'")
        except Exception as e:
            logger.debug(f"[Chat] RAG notice skipped: {e}")

    # ── Step 5: Invoke LLM ────────────────────────────────────────────────────
    response = _llm_with_tools.invoke(messages)

    if isinstance(response, AIMessage):
        if response.tool_calls:
            logger.info(f"[Chat] Tool(s): {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info(f"[Chat] Reply: {_extract_text(response.content)[:100]}")

    return {
        "messages":       [response],
        "critic_score":   state.get("critic_score", 0),
        "critic_retries": state.get("critic_retries", 0),
        "thread_id":      thread_id,
    }


def critic_node(state: State) -> State:
    """
    Quality verification — scores the last AI response 1-10.
    If score < 3 and retries < 2, sends feedback back to chat_node.
    """
    messages = state["messages"]
    retries  = state.get("critic_retries", 0)

    if retries >= 2:
        logger.info("[Critic] Max retries reached — passing through")
        return {"messages": [], "critic_score": state.get("critic_score", 0), "critic_retries": retries}

    last_ai = None
    for msg in reversed(messages):
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            text = _extract_text(msg.content)
            if text and len(text) > 30:
                last_ai = msg
                break

    if not last_ai:
        return {"messages": [], "critic_score": 0, "critic_retries": retries}

    last_human    = next((m for m in reversed(messages) if m.type == "human"), None)
    user_request  = _extract_text(last_human.content) if last_human else ""
    response_text = _extract_text(last_ai.content)

    critic_prompt = f"""You are a strict quality control agent reviewing an AI assistant's response.

USER REQUEST:
{user_request[:500]}

AI RESPONSE TO REVIEW:
{response_text[:2000]}

Score this response strictly on a scale of 1-10. Reply in EXACTLY this format, nothing else:
SCORE: [1-10]
VERDICT: PASS or RETRY
ISSUES: [comma-separated list of specific issues, or "none" if PASS]"""

    try:
        review      = _critic_llm.invoke(critic_prompt)
        review_text = _extract_text(review.content) if hasattr(review, "content") else ""

        score   = 10
        verdict = "PASS"
        issues  = "none"

        for line in review_text.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = int(line.split(":")[1].strip().split()[0])
                except Exception:
                    score = 10
            elif line.startswith("VERDICT:"):
                verdict = line.split(":")[1].strip().upper()
            elif line.startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()

        logger.info(f"[Critic] Score: {score}/10 | Verdict: {verdict} | Issues: {issues[:80]}")

        if verdict == "RETRY" and score < 3:
            feedback = HumanMessage(
                content=(
                    f"[QUALITY REVIEW — Score {score}/10]: Your response needs improvement.\n"
                    f"Issues: {issues}\n\n"
                    f"Please revise your answer to fix these specific issues."
                )
            )
            logger.info(f"[Critic] Revision requested (retry {retries + 1}/2)")
            return {
                "messages":       [feedback],
                "critic_score":   score,
                "critic_retries": retries + 1,
            }

        return {"messages": [], "critic_score": score, "critic_retries": retries}

    except Exception as e:
        logger.warning(f"[Critic] Review failed: {e} — passing through")
        return {"messages": [], "critic_score": 0, "critic_retries": retries}


def email_approval_node(state: State) -> State:
    """Human-in-the-loop gate for send_email."""
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state

    email_calls = [tc for tc in last_message.tool_calls if tc["name"] == "send_email"]
    if not email_calls:
        return state

    tc   = email_calls[0]
    args = tc["args"]

    human_response: str = interrupt({
        "type":         "email_approval",
        "message":      "An email is ready to send. Please review and approve or deny.",
        "to":           args.get("to", ""),
        "subject":      args.get("subject", ""),
        "body":         args.get("body", ""),
        "total_emails": len(email_calls),
        "instructions": 'Reply with "approve" to send or "deny" to cancel.',
    })

    decision = (human_response or "").strip().lower()

    if decision == "approve":
        logger.info(f"[Email] Approved → {args.get('to')}")
        return state

    logger.info(f"[Email] Denied — cancelling {len(email_calls)} email(s)")
    cancellations = [
        ToolMessage(
            tool_call_id=call["id"],
            name="send_email",
            content=(
                f"❌ Email to '{call['args'].get('to')}' was cancelled. "
                "Do not resend unless explicitly asked."
            ),
        )
        for call in email_calls
    ]
    return {"messages": cancellations}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_tools(state: State) -> str:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return "critic"
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return "critic"
    names = [tc["name"] for tc in tool_calls]
    logger.info(f"[Route] tools={names}")
    return "email_approval" if "send_email" in names else "tools"


def route_after_approval(state: State) -> str:
    return "chat" if isinstance(state["messages"][-1], ToolMessage) else "tools"


def route_after_critic(state: State) -> str:
    msgs = state["messages"]
    if msgs and isinstance(msgs[-1], HumanMessage):
        if "[QUALITY REVIEW" in str(msgs[-1].content):
            return "chat"
    return END


# ── Build ─────────────────────────────────────────────────────────────────────

def build_graph():
    """Synchronous build — used if you ever call this outside async context."""
    checkpointer = MemorySaver()
    return _compile(checkpointer)


async def build_graph_async():
    """
    Async build called from main.py lifespan.
    Uses MemorySaver — no database connection needed at all.
    """
    checkpointer = MemorySaver()
    chatbot      = _compile(checkpointer)
    logger.info(
        "[Graph] Compiled with MemorySaver ✓\n"
        "        Memory: sliding-window summarization (MEMORY_MAX_MESSAGES / MEMORY_KEEP_LAST_K)\n"
        "        Summary storage: SQLite summaries table\n"
        "        Conversation history lives in-process (survives chat, lost on restart).\n"
        "        Chat history for /history endpoint is persisted in SQLite via database.py."
    )
    return chatbot


def _compile(checkpointer):
    graph = StateGraph(State)

    graph.add_node("chat",           chat_node)
    graph.add_node("email_approval", email_approval_node)
    graph.add_node("tools",          ToolNode(TOOLS))
    graph.add_node("critic",         critic_node)

    graph.set_entry_point("chat")

    graph.add_conditional_edges(
        "chat", route_tools,
        {"email_approval": "email_approval", "tools": "tools", "critic": "critic"},
    )
    graph.add_conditional_edges(
        "email_approval", route_after_approval,
        {"tools": "tools", "chat": "chat"},
    )
    graph.add_edge("tools", "chat")
    graph.add_conditional_edges(
        "critic", route_after_critic,
        {"chat": "chat", END: END},
    )

    return graph.compile(checkpointer=checkpointer)


async def shutdown_checkpointer() -> None:
    """Called from main.py lifespan shutdown. Nothing to close for MemorySaver."""
    from agent.database import close_db
    await close_db()
    logger.info("[Graph] Shutdown complete")


# Expose compiled chatbot for direct import if needed
chatbot = build_graph()
logger.info(
    "LangGraph Multi-Agent System compiled ✓\n"
    "  Agents : chat_node (primary) + critic_node (verifier) + email_approval (HITL)\n"
    "  Memory : MemorySaver (in-process) + SQLite (chat log + summaries)\n"
    "  Window : sliding-window summarization prevents token overflow\n"
    "  Options: 1=workflow orchestration  2=meeting intelligence  3=critic agent"
)
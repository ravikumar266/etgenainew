"""
graph.py — LangGraph multi-agent orchestration

Fix 2 applied: Critic node REMOVED from default flow.
  - Was burning 1 extra API call per message (50% quota waste)
  - Quality check now available via /feedback endpoint only
  - Flow: chat_node → tools → chat_node → END

Fix 5+6 applied via tools_rag.py and memory_manager.py:
  - ChromaDB client cached (no re-init per request)
  - Summary cached in memory (no repeated SQLite reads)
"""

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.config import llm, logger
from agent.prompts import SYSTEM_PROMPT
from agent.memory_manager import maybe_summarize, inject_summary_context, estimate_message_tokens

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

# RAG notice cache — checked once per thread, not every message
_rag_checked: set = set()


class State(TypedDict):
    messages:   Annotated[list[AnyMessage], add_messages]
    thread_id:  Optional[str]


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text","") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _get_thread_id(config: RunnableConfig) -> str:
    try:
        cfg = config if isinstance(config, dict) else {}
        return cfg.get("configurable", {}).get("thread_id", "default")
    except Exception:
        return "default"


def _has_rag_notice(messages: list) -> bool:
    return any(
        "[RAG CONTEXT AVAILABLE]" in _extract_text(m.content)
        for m in messages if isinstance(m, HumanMessage)
    )


async def chat_node(state: State, config: RunnableConfig) -> State:
    """
    Primary agent — Gemini with all tools.
    No critic call. One LLM call per user message.
    """
    thread_id = _get_thread_id(config)
    messages  = list(state["messages"])

    # Step 1: Memory — summarize + trim (uses cached summary)
    token_est = estimate_message_tokens(messages)
    logger.info(f"[Chat] '{thread_id}': {len(messages)} msgs ≈ {token_est} tokens")
    messages = await maybe_summarize(thread_id, messages, llm)

    # Step 2: Inject cached summary
    messages = await inject_summary_context(thread_id, messages)

    # Step 3: System message first
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    # Step 4: RAG notice — ONCE per thread (cached, no repeated DB calls)
    if thread_id not in _rag_checked and not _has_rag_notice(messages):
        try:
            from agent.tools_rag import _get_chroma_client
            client     = _get_chroma_client()
            col_names  = [c.name for c in client.list_collections()]
            safe_name  = thread_id.lower()[:63]
            has_docs   = any(safe_name in n or n == "default" for n in col_names)

            if has_docs:
                messages = messages[:1] + [
                    HumanMessage(content=(
                        f"[RAG CONTEXT AVAILABLE] Docs in collection '{thread_id}'. "
                        f"Use query_rag when user asks about uploaded docs."
                    )),
                    AIMessage(content="Understood. I'll query RAG when needed."),
                ] + messages[1:]
                logger.info(f"[Chat] RAG notice injected for '{thread_id}'")
        except Exception as e:
            logger.debug(f"[Chat] RAG check failed: {e}")
        finally:
            _rag_checked.add(thread_id)

    # Step 5: Single LLM call — no critic after this
    response = _llm_with_tools.invoke(messages)

    if isinstance(response, AIMessage):
        if response.tool_calls:
            logger.info(f"[Chat] Tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info(f"[Chat] Reply: {_extract_text(response.content)[:80]}")

    return {
        "messages":  [response],
        "thread_id": thread_id,
    }


def email_approval_node(state: State) -> State:
    """Human-in-the-loop gate for send_email."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return state
    email_calls = [tc for tc in last.tool_calls if tc["name"] == "send_email"]
    if not email_calls:
        return state

    args = email_calls[0]["args"]
    human_response: str = interrupt({
        "type":    "email_approval",
        "to":      args.get("to", ""),
        "subject": args.get("subject", ""),
        "body":    args.get("body", ""),
    })

    if (human_response or "").strip().lower() == "approve":
        logger.info(f"[Email] Approved → {args.get('to')}")
        return state

    logger.info("[Email] Denied")
    return {"messages": [
        ToolMessage(
            tool_call_id=call["id"], name="send_email",
            content=f"❌ Email to '{call['args'].get('to')}' cancelled."
        ) for call in email_calls
    ]}


def route_after_chat(state: State) -> str:
    """Route: tool call → tools, else → END. No critic."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return END
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    names = [tc["name"] for tc in tool_calls]
    logger.info(f"[Route] tools={names}")
    return "email_approval" if "send_email" in names else "tools"


def route_after_approval(state: State) -> str:
    return "chat" if isinstance(state["messages"][-1], ToolMessage) else "tools"


def build_graph():
    return _compile(MemorySaver())


async def build_graph_async():
    chatbot = _compile(MemorySaver())
    logger.info(
        "[Graph] Compiled ✓\n"
        "  Critic:  REMOVED — 1 LLM call per message now\n"
        "  RAG:     Cached client + cached per-thread notice\n"
        "  Memory:  In-memory summary cache (no repeated SQLite reads)\n"
        "  Flow:    chat → tools → chat → END"
    )
    return chatbot


def _compile(checkpointer):
    graph = StateGraph(State)

    graph.add_node("chat",           chat_node)
    graph.add_node("email_approval", email_approval_node)
    graph.add_node("tools",          ToolNode(TOOLS))

    graph.set_entry_point("chat")

    graph.add_conditional_edges(
        "chat", route_after_chat,
        {"email_approval": "email_approval", "tools": "tools", END: END}
    )
    graph.add_conditional_edges(
        "email_approval", route_after_approval,
        {"tools": "tools", "chat": "chat"}
    )
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=checkpointer)


async def shutdown_checkpointer() -> None:
    from agent.database import close_db
    await close_db()
    logger.info("[Graph] Shutdown complete")


chatbot = build_graph()
logger.info(
    "LangGraph compiled ✓\n"
    "  1 LLM call per message (critic removed)\n"
    "  ChromaDB cached | Summary cached | RAG notice cached"
)
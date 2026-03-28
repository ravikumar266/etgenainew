"""
main.py — ETGenAI v3
────────────────────
Memory architecture (correct):
  AsyncPostgresSaver is the SINGLE source of truth for conversation state.
  LangGraph loads prior state automatically per thread_id on every invoke().
  /chat sends ONLY the current HumanMessage — no manual history injection.
  Custom chats/summaries tables are kept for /history and /debug endpoints only.

RAG endpoints (full pipeline):
  POST /upload-pdf              — upload + ingest a PDF into ChromaDB
  POST /rag/ingest-url          — ingest a webpage URL into ChromaDB
  POST /rag/query               — query any RAG collection → LLM-synthesised answer
  GET  /rag/collections         — list all collections + chunk counts
  DELETE /rag/collections/{name}— delete a collection

Typical RAG usage:
  1. Upload a PDF:
       curl -X POST /upload-pdf -F "file=@report.pdf" -F "collection=my-report"
  2. Query it:
       curl -X POST /rag/query \
            -H "Content-Type: application/json" \
            -d '{"question":"What are the key findings?","collection":"my-report"}'
"""

import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.config import logger
from agent.database import (
    debug_thread,
    init_db,
    list_threads,
    load_messages,
    load_summary,
    save_message,
    save_summary,
)
from agent.scheduler import get_scheduler, start_scheduler, stop_scheduler


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    from agent.graph import build_graph_async, shutdown_checkpointer
    app.state.chatbot = await build_graph_async()
    logger.info("[App] Chatbot ready")

    start_scheduler()
    yield
    stop_scheduler()

    await shutdown_checkpointer()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ETGenAI",
    description="Autonomous AI Agent — Multi-key Gemini + Groq fallback + RAG",
    version="3.2.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_pending_emails: dict = {}
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:   str
    thread_id: str = ""


class EmailApprovalRequest(BaseModel):
    thread_id: str
    decision:  str   # "approve" | "deny"


class RAGQueryRequest(BaseModel):
    """
    Body for POST /rag/query.
    - question   : natural-language question to answer from the knowledge base
    - collection : must match the collection name used during ingest (default: "default")
    - top_k      : how many chunks to retrieve before synthesising (1-10, default: 4)
    """
    question:   str
    collection: str = "default"
    top_k:      int = 4


class RAGIngestURLRequest(BaseModel):
    """Body for POST /rag/ingest-url."""
    url:        str
    collection: str = "default"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_reply(messages: list) -> str:
    for msg in reversed(messages):
        if msg.type != "ai":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                return text
    return ""


def _extract_tools_used(messages: list) -> list:
    seen = []
    for msg in messages:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                if name and name not in seen:
                    seen.append(name)
    return seen


def _extract_pending_email(messages: list) -> dict | None:
    for msg in reversed(messages):
        if msg.type != "ai":
            continue
        for tc in getattr(msg, "tool_calls", []):
            if tc.get("name") == "send_email":
                args = tc.get("args", {})
                return {
                    "to":      args.get("to", ""),
                    "subject": args.get("subject", ""),
                    "body":    args.get("body", ""),
                }
    return None


def _llm_content_to_str(content) -> str:
    """Safely extract plain text from Gemini or Groq response content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content).strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "ETGenAI v3.2 running"})


# ── CHAT ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphInterrupt

    thread_id = req.thread_id.strip() or str(uuid.uuid4())
    chatbot   = request.app.state.chatbot

    logger.info(f"[API] /chat thread={thread_id} msg={req.message[:80]}")

    await save_message(thread_id, "user", req.message)

    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "messages":  [HumanMessage(content=req.message)],
        "thread_id": thread_id,
        "summary":   "",
    }

    try:
        state = await _invoke_async(chatbot, initial_input, config)

    except GraphInterrupt as gi:
        payload = {}
        try:
            interrupts = gi.args[0] if gi.args else []
            if interrupts:
                payload = getattr(interrupts[0], "value", {})
        except Exception:
            pass

        pending = {
            "to":      payload.get("to", ""),
            "subject": payload.get("subject", ""),
            "body":    payload.get("body", ""),
        }
        _pending_emails[thread_id] = pending
        reply_text = (
            f"📧 Email ready for {pending['to']} — \"{pending['subject']}\". "
            "POST /email/approve to send."
        )
        await save_message(thread_id, "ai", reply_text)
        return JSONResponse({
            "reply":                   reply_text,
            "thread_id":               thread_id,
            "tools_used":              [],
            "pending_email":           pending,
            "email_approval_required": True,
        })

    except Exception as e:
        logger.error(f"[API] Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    messages    = state.get("messages", [])
    reply       = _extract_reply(messages)
    tools_used  = _extract_tools_used(messages)
    new_summary = state.get("summary", "")

    if reply:
        await save_message(thread_id, "ai", reply)
    if new_summary:
        old_summary = await load_summary(thread_id)
        if new_summary != old_summary:
            await save_summary(thread_id, new_summary)

    pending_email           = _extract_pending_email(messages)
    email_approval_required = False
    if pending_email:
        _pending_emails[thread_id] = pending_email
        email_approval_required    = True

    return JSONResponse({
        "reply":                   reply,
        "thread_id":               thread_id,
        "tools_used":              tools_used,
        "pending_email":           pending_email,
        "email_approval_required": email_approval_required,
    })


async def _invoke_async(chatbot, initial_input: dict, config: dict) -> dict:
    import asyncio
    if hasattr(chatbot, "ainvoke"):
        return await chatbot.ainvoke(initial_input, config=config)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: chatbot.invoke(initial_input, config=config)
    )


# ── THREAD HISTORY ────────────────────────────────────────────────────────────

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    messages = await load_messages(thread_id)
    summary  = await load_summary(thread_id)
    if not messages and not summary:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    return JSONResponse({
        "thread_id": thread_id,
        "summary":   summary,
        "messages":  messages,
        "count":     len(messages),
    })


@app.get("/threads")
async def get_threads():
    threads = await list_threads()
    return JSONResponse({"threads": threads, "total": len(threads)})


@app.post("/threads/new")
async def new_thread():
    return JSONResponse({"thread_id": str(uuid.uuid4())})


# ── DEBUG ─────────────────────────────────────────────────────────────────────

@app.get("/debug/memory/{thread_id}")
async def debug_memory(thread_id: str):
    result = await debug_thread(thread_id)
    result["checkpoint_note"] = (
        "LangGraph state lives in 'langgraph_checkpoints' (AsyncPostgresSaver). "
        "The 'chats' table is only used for /history and /debug."
    )
    return JSONResponse(result)


@app.get("/debug/sql-verify")
async def debug_sql_verify():
    return JSONResponse({
        "custom_chats_table": {
            "all_recent":       "SELECT thread_id, role, LEFT(content,80), created_at FROM chats ORDER BY created_at DESC LIMIT 20;",
            "by_thread":        "SELECT role, LEFT(content,100), created_at FROM chats WHERE thread_id='YOUR_ID' ORDER BY created_at;",
            "count_per_thread": "SELECT thread_id, COUNT(*) FROM chats GROUP BY thread_id ORDER BY COUNT(*) DESC;",
        },
        "langgraph_checkpoint_tables": {
            "list_threads": "SELECT DISTINCT thread_id FROM langgraph_checkpoints;",
            "thread_state": "SELECT thread_id, checkpoint_id, created_at FROM langgraph_checkpoints WHERE thread_id='YOUR_ID' ORDER BY created_at;",
            "all_tables":   "SELECT tablename FROM pg_tables WHERE schemaname='public';",
        },
        "summary_table": {"all": "SELECT thread_id, LEFT(summary,200), updated_at FROM summaries;"},
        "connect": "docker exec -it etgenai_postgres psql -U chatuser -d chatdb",
    })


# ── EMAIL APPROVAL ────────────────────────────────────────────────────────────

@app.post("/email/approve")
async def approve_email(req: EmailApprovalRequest, request: Request):
    decision = req.decision.lower().strip()
    if decision not in ("approve", "deny"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'deny'")

    pending = _pending_emails.get(req.thread_id)
    if not pending:
        raise HTTPException(status_code=404, detail=f"No pending email for thread '{req.thread_id}'")
    del _pending_emails[req.thread_id]

    chatbot = request.app.state.chatbot
    config  = {"configurable": {"thread_id": req.thread_id}}

    if decision == "deny":
        logger.info(f"[Email] DENIED thread={req.thread_id}")
        try:
            from langgraph.types import Command
            await _invoke_async(chatbot, Command(resume="deny"), config)
        except Exception:
            pass
        return JSONResponse({"success": True, "decision": "denied", "message": "Email cancelled."})

    logger.info(f"[Email] APPROVED → {pending['to']} thread={req.thread_id}")
    try:
        from langgraph.types import Command
        await _invoke_async(chatbot, Command(resume="approve"), config)
        return JSONResponse({
            "success":  True,
            "decision": "approved",
            "message":  f"Email sent to {pending['to']}",
            "email":    {"to": pending["to"], "subject": pending["subject"]},
        })
    except Exception as e:
        logger.error(f"[Email] Resume failed: {e} — direct send fallback")
        try:
            from agent.tools_email import _send_email_direct
            _send_email_direct(pending["to"], pending["subject"], pending["body"])
            return JSONResponse({"success": True, "decision": "approved",
                                 "message": f"Sent to {pending['to']} (direct fallback)"})
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Send failed: {e2}")


@app.get("/email/pending/{thread_id}")
async def get_pending_email(thread_id: str):
    pending = _pending_emails.get(thread_id)
    if not pending:
        return JSONResponse({"has_pending": False, "pending_email": None})
    return JSONResponse({"has_pending": True, "pending_email": pending})


# ══════════════════════════════════════════════════════════════════════════════
# RAG  —  Full pipeline: ingest → store → query → answer
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Upload + ingest a PDF ──────────────────────────────────────────────────

@app.post("/upload-pdf")
async def upload_pdf(
    file:       UploadFile = File(..., description="PDF file to ingest"),
    collection: str        = Form(default="default", description="RAG collection name"),
):
    """
    Upload a PDF and ingest it into ChromaDB for RAG querying.

    multipart/form-data fields:
      file       — the .pdf binary
      collection — collection name to store chunks in  (default: "default")

    After this call succeeds, query the PDF with:
      POST /rag/query  { "question": "...", "collection": "<same name>" }

    Returns:
      { success, message, chunks, collection, filename, size_kb }
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only .pdf files are supported. Received: '{filename}'"
        )

    tmp_path: str | None = None
    try:
        # Stream to disk in 64 KB chunks — avoids loading the whole file into RAM
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix="etgenai_upload_"
        ) as tmp:
            tmp_path    = tmp.name
            total_bytes = 0
            read_size   = 64 * 1024

            while True:
                chunk = await file.read(read_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_PDF_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: {total_bytes / 1_048_576:.1f} MB "
                            f"(max {_MAX_PDF_BYTES // 1_048_576} MB)"
                        ),
                    )
                tmp.write(chunk)

        logger.info(
            f"[PDF Upload] '{filename}' ({total_bytes / 1024:.1f} KB) "
            f"→ temp={tmp_path} collection='{collection}'"
        )

        from agent.tools_rag import ingest_pdf
        result: str = ingest_pdf.invoke({"file_path": tmp_path, "collection": collection})

        chunks = 0
        m = re.search(r"(\d+)\s+chunk", result)
        if m:
            chunks = int(m.group(1))

        if any(w in result.lower() for w in ["error", "failed", "not found", "no readable"]):
            logger.warning(f"[PDF Upload] Ingest issue: {result}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": result, "chunks": 0,
                         "collection": collection, "filename": filename},
            )

        logger.info(f"[PDF Upload] ✓ '{filename}' → {chunks} chunks in '{collection}'")
        return JSONResponse({
            "success":    True,
            "message":    (
                f"'{filename}' ingested into collection '{collection}' ({chunks} chunks). "
                f"Query it with: POST /rag/query "
                f'{{ "question": "...", "collection": "{collection}" }}'
            ),
            "chunks":     chunks,
            "collection": collection,
            "filename":   filename,
            "size_kb":    round(total_bytes / 1024, 1),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PDF Upload] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as ce:
                logger.warning(f"[PDF Upload] Temp file cleanup failed: {ce}")


# ── 2. Ingest a webpage URL ───────────────────────────────────────────────────

@app.post("/rag/ingest-url")
async def rag_ingest_url(req: RAGIngestURLRequest):
    """
    Fetch a webpage and ingest its text content into ChromaDB.

    Body (JSON):
      { "url": "https://example.com/article", "collection": "my-docs" }

    After ingesting, query with:
      POST /rag/query  { "question": "...", "collection": "my-docs" }
    """
    if not req.url.startswith("http"):
        raise HTTPException(
            status_code=400,
            detail="url must start with http:// or https://"
        )

    try:
        from agent.tools_rag import ingest_webpage
        result: str = ingest_webpage.invoke({"url": req.url, "collection": req.collection})

        chunks = 0
        m = re.search(r"(\d+)\s+chunk", result)
        if m:
            chunks = int(m.group(1))

        if any(w in result.lower() for w in ["error", "failed", "could not", "timeout"]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": result, "chunks": 0,
                         "collection": req.collection, "url": req.url},
            )

        logger.info(f"[RAG Ingest URL] ✓ '{req.url}' → {chunks} chunks in '{req.collection}'")
        return JSONResponse({
            "success":    True,
            "message":    (
                f"'{req.url}' ingested into '{req.collection}' ({chunks} chunks). "
                f"Query it with: POST /rag/query "
                f'{{ "question": "...", "collection": "{req.collection}" }}'
            ),
            "chunks":     chunks,
            "collection": req.collection,
            "url":        req.url,
        })

    except Exception as e:
        logger.error(f"[RAG Ingest URL] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Query a RAG collection → LLM-synthesised answer ───────────────────────

@app.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    """
    Ask a question against any RAG collection and get a synthesised answer.

    Body (JSON):
      {
        "question":   "What are the key findings in chapter 3?",
        "collection": "my-report",   // same name used during ingest
        "top_k":      4              // chunks to retrieve (1–10)
      }

    How it works:
      1. Semantic search in ChromaDB — finds the top_k most relevant chunks
      2. Passes the chunks + question to the LLM (Gemini / Groq fallback)
      3. LLM synthesises a grounded answer, citing which chunks support each point

    Returns:
      {
        "success":    true,
        "question":   "...",
        "collection": "my-report",
        "answer":     "<LLM synthesised answer with citations>",
        "raw_chunks": "<full retrieved text for debugging>",
        "chunk_refs": ["[1] PDF | ...", ...],
        "top_k":      4
      }
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    top_k = max(1, min(req.top_k, 10))

    try:
        # ── Step 1: Retrieve relevant chunks from ChromaDB ────────────────────
        from agent.tools_rag import query_rag
        raw_chunks: str = query_rag.invoke({
            "question":   req.question,
            "collection": req.collection,
            "top_k":      top_k,
        })

        # Return early if collection is empty or no results found
        no_data_signals = ["is empty", "No relevant results", "query failed", "add content first"]
        if any(signal in raw_chunks for signal in no_data_signals):
            return JSONResponse({
                "success":    False,
                "question":   req.question,
                "collection": req.collection,
                "answer":     raw_chunks,
                "raw_chunks": raw_chunks,
                "chunk_refs": [],
                "top_k":      top_k,
                "hint": (
                    f"Collection '{req.collection}' appears to be empty. "
                    "Ingest content first with POST /upload-pdf or POST /rag/ingest-url"
                ),
            })

        # ── Step 2: LLM synthesises a grounded answer ─────────────────────────
        from agent.config import llm

        synthesis_prompt = f"""You are a precise research assistant. Your job is to answer
the user's question using ONLY the document chunks provided below.

Rules:
- Answer clearly and in full sentences
- Cite the chunk number(s) that support each key point, e.g. [chunk 1]
- If a piece of information comes from a specific source (PDF name, URL),
  mention it naturally
- If the chunks do not contain enough information to fully answer the question,
  say so explicitly — do NOT invent or assume facts
- Keep the answer focused and well-structured

USER QUESTION:
{req.question}

RETRIEVED DOCUMENT CHUNKS:
{raw_chunks}

Provide your synthesised answer below:"""

        response      = llm.invoke(synthesis_prompt)
        answer        = _llm_content_to_str(response.content)

        # ── Step 3: Extract chunk reference lines for the response ─────────────
        chunk_refs = [
            line.strip()
            for line in raw_chunks.split("\n")
            if line.strip().startswith("[") and "|" in line
        ]

        logger.info(
            f"[RAG Query] ✓ question='{req.question[:60]}' "
            f"collection='{req.collection}' chunks_retrieved={len(chunk_refs)}"
        )

        return JSONResponse({
            "success":    True,
            "question":   req.question,
            "collection": req.collection,
            "answer":     answer,
            "raw_chunks": raw_chunks,
            "chunk_refs": chunk_refs,
            "top_k":      top_k,
        })

    except Exception as e:
        logger.error(f"[RAG Query] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


# ── 4. List / delete collections ─────────────────────────────────────────────

@app.get("/rag/collections")
async def get_rag_collections():
    """
    List all RAG collections with their chunk counts.

    Use this to confirm a PDF or URL was ingested successfully
    before running a query.
    """
    try:
        import chromadb
        from agent.tools_rag import CHROMA_DIR
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        cols   = client.list_collections()
        return JSONResponse({
            "collections": [{"name": c.name, "chunks": c.count()} for c in cols],
            "total": len(cols),
            "endpoints": {
                "ingest_pdf": "POST /upload-pdf          (form: file + collection)",
                "ingest_url": "POST /rag/ingest-url      (json: url + collection)",
                "query":      "POST /rag/query           (json: question + collection + top_k)",
                "delete":     "DELETE /rag/collections/{name}",
            },
        })
    except Exception as e:
        return JSONResponse({"collections": [], "total": 0, "error": str(e)})


@app.delete("/rag/collections/{name}")
async def delete_collection(name: str):
    """Permanently delete a RAG collection and all its stored chunks."""
    try:
        from agent.tools_rag import delete_rag_collection
        result = delete_rag_collection.invoke({"collection": name})
        return JSONResponse({"success": True, "message": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from agent.config import _GEMINI_KEYS, _GEMINI_MODEL, _GROQ_MODEL, _GROQ_API_KEY
    sched = get_scheduler()
    return JSONResponse({
        "status":         "ok",
        "version":        "3.2.0",
        "checkpointer":   "AsyncPostgresSaver",
        "scheduler":      sched.running if sched else False,
        "pending_emails": len(_pending_emails),
        "llm": {
            "gemini_model":  _GEMINI_MODEL,
            "gemini_keys":   len(_GEMINI_KEYS),
            "groq_fallback": _GROQ_MODEL if _GROQ_API_KEY else "disabled",
        },
        "rag": {
            "ingest_pdf": "POST /upload-pdf",
            "ingest_url": "POST /rag/ingest-url",
            "query":      "POST /rag/query",
            "list":       "GET  /rag/collections",
            "delete":     "DELETE /rag/collections/{name}",
        },
    })


@app.get("/scheduler/status")
async def scheduler_status():
    sched = get_scheduler()
    if not sched:
        return JSONResponse({"running": False, "jobs": []})
    return JSONResponse({
        "running": sched.running,
        "jobs": [
            {"id": j.id, "name": j.name,
             "next_run": str(j.next_run_time) if j.next_run_time else None}
            for j in sched.get_jobs()
        ],
    })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
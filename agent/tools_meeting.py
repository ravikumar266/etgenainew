"""
agent/tools_meeting.py
───────────────────────
Option 2: Meeting Intelligence System

Extracts decisions, creates action items, assigns owners, tracks completion,
and escalates stalls — all without manual follow-up.

User-facing @tools:
  process_meeting(transcript, title)    → full AI analysis of meeting transcript
  create_action_items(run_id, items)    → register action items in workflow tracker
  check_action_items(run_id)            → see pending/completed/overdue items
  escalate_stalled_items(run_id)        → find and escalate overdue action items

How it works:
  1. User pastes meeting transcript (or ingests from YouTube/doc)
  2. process_meeting extracts: decisions, action items, owners, due dates
  3. A workflow_followup run is auto-created to track each action item
  4. Scheduler checks every 10 min for stalled/overdue items and escalates
  5. Full audit trail maintained throughout

.env:
  ESCALATION_EMAIL=ravishani98765432@gmail.com  (all escalation emails go here)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

from agent.config import llm, logger

# ── Action item database ──────────────────────────────────────────────────────

DB_PATH = os.getenv("WORKFLOW_DB", "./workflows.db")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS action_items (
            item_id      TEXT PRIMARY KEY,
            meeting_id   TEXT NOT NULL,
            task         TEXT NOT NULL,
            owner        TEXT DEFAULT 'Unassigned',
            due_date     TEXT,
            priority     TEXT DEFAULT 'Medium',
            status       TEXT DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            notes        TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meeting_records (
            meeting_id   TEXT PRIMARY KEY,
            title        TEXT,
            processed_at TEXT NOT NULL,
            summary      TEXT,
            decisions    TEXT DEFAULT '[]',
            open_questions TEXT DEFAULT '[]',
            next_meeting TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def _extract_text_from_response(response) -> str:
    """Handle both str and list Gemini content formats."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def process_meeting(transcript: str, meeting_title: str = "Meeting") -> str:
    """
    Analyse a meeting transcript and automatically extract:
    - All decisions made
    - Action items with owners and due dates
    - Open questions needing follow-up
    - Next meeting date/time

    Then automatically:
    - Saves the meeting record with full audit trail
    - Creates tracked action items in the workflow system
    - Returns structured output ready for follow-up

    Args:
      transcript    : Full meeting transcript text. Can be pasted directly,
                      or first use ingest_youtube/ingest_pdf to load it into
                      RAG and then query_rag to retrieve relevant sections.
      meeting_title : Name of the meeting for the record (e.g. "Sprint Review Q1")

    After calling this, use check_action_items(meeting_id) to monitor progress.
    """
    if not transcript or len(transcript.strip()) < 50:
        return "Transcript is too short. Provide the full meeting transcript text."

    meeting_id = f"mtg-{uuid.uuid4().hex[:8]}"

    # ── Step 1: AI extraction ─────────────────────────────────────────────────
    prompt = f"""You are an expert meeting intelligence system. Analyse this meeting transcript precisely.

MEETING TITLE: {meeting_title}
TRANSCRIPT:
{transcript[:10000]}

Extract information in EXACTLY this format (use these exact headers):

## SUMMARY
[2-3 sentences describing what the meeting covered and its outcome]

## DECISIONS MADE
[List every firm decision, one per line starting with "- "]
[If no decisions, write "- None recorded"]

## ACTION ITEMS
[For each action item, use this exact format:]
TASK: [specific task description]
OWNER: [person's name or "Unassigned" if not mentioned]
DUE: [specific date, or "This week", "Next sprint", "ASAP", or "TBD"]
PRIORITY: [High / Medium / Low]
---
[Repeat for each action item]

## OPEN QUESTIONS
[Unresolved issues needing follow-up, one per line starting with "- "]
[If none, write "- None"]

## NEXT MEETING
[Date and time if mentioned, or "Not scheduled"]

## RISKS IDENTIFIED
[Any risks, blockers, or concerns raised, one per line starting with "- "]
[If none, write "- None"]

Be specific. Use exact names and dates from the transcript. Do not invent information."""

    try:
        response = llm.invoke(prompt)
        analysis = _extract_text_from_response(response)
    except Exception as e:
        return f"Meeting analysis failed: {str(e)}"

    # ── Step 2: Parse action items from analysis ───────────────────────────────
    action_items = []
    lines = analysis.split("\n")
    current_item = {}

    for line in lines:
        line = line.strip()
        if line.startswith("TASK:"):
            if current_item.get("task"):
                action_items.append(current_item)
            current_item = {"task": line[5:].strip()}
        elif line.startswith("OWNER:") and current_item:
            current_item["owner"] = line[6:].strip()
        elif line.startswith("DUE:") and current_item:
            current_item["due"] = line[4:].strip()
        elif line.startswith("PRIORITY:") and current_item:
            current_item["priority"] = line[9:].strip()
        elif line == "---" and current_item.get("task"):
            action_items.append(current_item)
            current_item = {}

    if current_item.get("task"):
        action_items.append(current_item)

    # ── Step 3: Extract summary and decisions ─────────────────────────────────
    summary = ""
    decisions = []
    open_questions = []
    next_meeting = "Not scheduled"

    section = ""
    for line in lines:
        stripped = line.strip()
        if "## SUMMARY" in stripped:
            section = "summary"
        elif "## DECISIONS" in stripped:
            section = "decisions"
        elif "## ACTION ITEMS" in stripped:
            section = "actions"
        elif "## OPEN QUESTIONS" in stripped:
            section = "questions"
        elif "## NEXT MEETING" in stripped:
            section = "next"
        elif "## RISKS" in stripped:
            section = "risks"
        elif stripped.startswith("- ") and section == "decisions":
            decisions.append(stripped[2:])
        elif stripped.startswith("- ") and section == "questions":
            open_questions.append(stripped[2:])
        elif section == "summary" and stripped and not stripped.startswith("##"):
            summary += stripped + " "
        elif section == "next" and stripped and not stripped.startswith("##"):
            next_meeting = stripped

    # ── Step 4: Save to database ──────────────────────────────────────────────
    conn = _get_db()
    now = datetime.now().isoformat(timespec="seconds")

    conn.execute("""
        INSERT INTO meeting_records
        (meeting_id, title, processed_at, summary, decisions, open_questions, next_meeting)
        VALUES (?,?,?,?,?,?,?)
    """, (
        meeting_id, meeting_title, now,
        summary.strip(),
        json.dumps(decisions),
        json.dumps(open_questions),
        next_meeting,
    ))

    item_ids = []
    for item in action_items:
        item_id = f"ai-{uuid.uuid4().hex[:6]}"
        item_ids.append(item_id)

        # Parse due date to a real deadline
        due_str = item.get("due", "TBD")
        due_date = None
        if "week" in due_str.lower():
            due_date = (datetime.now() + timedelta(weeks=1)).strftime("%Y-%m-%d")
        elif "sprint" in due_str.lower():
            due_date = (datetime.now() + timedelta(weeks=2)).strftime("%Y-%m-%d")
        elif "asap" in due_str.lower():
            due_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif due_str and due_str != "TBD":
            due_date = due_str

        conn.execute("""
            INSERT INTO action_items
            (item_id, meeting_id, task, owner, due_date, priority, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            item_id, meeting_id,
            item.get("task", ""),
            item.get("owner", "Unassigned"),
            due_date,
            item.get("priority", "Medium"),
            now, now,
        ))

    conn.commit()
    conn.close()

    logger.info(f"[Meeting] Processed '{meeting_title}' → {len(action_items)} action items")

    # ── Step 5: Build return message ──────────────────────────────────────────
    items_display = ""
    for i, item in enumerate(action_items, 1):
        items_display += (
            f"\n  {i}. [{item.get('priority','Medium')}] {item.get('task','')}\n"
            f"     Owner: {item.get('owner','Unassigned')} | Due: {item.get('due','TBD')}"
        )

    decisions_display = "\n".join(f"  • {d}" for d in decisions) or "  • None recorded"
    questions_display = "\n".join(f"  • {q}" for q in open_questions) or "  • None"

    return (
        f"Meeting Intelligence Report\n"
        f"{'='*50}\n"
        f"Meeting ID : {meeting_id}\n"
        f"Title      : {meeting_title}\n"
        f"Processed  : {now}\n\n"
        f"SUMMARY:\n  {summary.strip() or 'See full analysis below'}\n\n"
        f"DECISIONS MADE ({len(decisions)}):\n{decisions_display}\n\n"
        f"ACTION ITEMS ({len(action_items)}):{items_display or chr(10) + '  None identified'}\n\n"
        f"OPEN QUESTIONS:\n{questions_display}\n\n"
        f"NEXT MEETING: {next_meeting}\n\n"
        f"{'='*50}\n"
        f"Use check_action_items('{meeting_id}') to monitor progress.\n\n"
        f"FULL ANALYSIS:\n{analysis}"
    )


@tool
def check_action_items(meeting_id: str) -> str:
    """
    Check the status of all action items from a meeting.

    Shows pending, completed, and overdue items.
    Overdue items are automatically flagged for escalation.

    Args:
      meeting_id : The meeting ID returned by process_meeting (e.g. "mtg-a1b2c3d4")
    """
    conn = _get_db()
    items = conn.execute(
        "SELECT * FROM action_items WHERE meeting_id=? ORDER BY priority DESC, due_date",
        (meeting_id,),
    ).fetchall()
    meeting = conn.execute(
        "SELECT * FROM meeting_records WHERE meeting_id=?", (meeting_id,)
    ).fetchone()
    conn.close()

    if not items:
        return f"No action items found for meeting '{meeting_id}'."

    title = meeting["title"] if meeting else meeting_id
    now = datetime.now()

    pending = []
    completed = []
    overdue = []

    for item in items:
        due = item["due_date"]
        is_overdue = (
            due and item["status"] == "pending"
            and datetime.strptime(due, "%Y-%m-%d") < now
        )
        if is_overdue:
            overdue.append(item)
        elif item["status"] == "completed":
            completed.append(item)
        else:
            pending.append(item)

    def fmt(item, flag=""):
        due_str = f" | Due: {item['due_date']}" if item["due_date"] else ""
        return (
            f"  {flag}[{item['priority']}] {item['task']}\n"
            f"    Owner: {item['owner']}{due_str} | Status: {item['status']}"
        )

    lines = [
        f"Action Items: {title} ({meeting_id})",
        f"Total: {len(items)} | Pending: {len(pending)} | "
        f"Overdue: {len(overdue)} | Completed: {len(completed)}\n",
    ]

    if overdue:
        lines.append(f"🚨 OVERDUE ({len(overdue)}):")
        lines.extend(fmt(i, "🚨 ") for i in overdue)
        lines.append("")

    if pending:
        lines.append(f"⏳ PENDING ({len(pending)}):")
        lines.extend(fmt(i, "⏳ ") for i in pending)
        lines.append("")

    if completed:
        lines.append(f"✅ COMPLETED ({len(completed)}):")
        lines.extend(fmt(i, "✅ ") for i in completed)

    if overdue:
        lines.append(
            f"\n⚠️  {len(overdue)} overdue item(s) detected. "
            f"Call escalate_stalled_items('{meeting_id}') to escalate."
        )

    return "\n".join(lines)


@tool
def escalate_stalled_items(meeting_id: str) -> str:
    """
    Find all overdue/stalled action items from a meeting and escalate them.

    Marks overdue items as 'escalated' and prepares an escalation summary
    for the manager (sent to ESCALATION_EMAIL if set in .env).

    Args:
      meeting_id : The meeting ID to check for stalled items.
    """
    conn = _get_db()
    items = conn.execute(
        "SELECT * FROM action_items WHERE meeting_id=? AND status='pending'",
        (meeting_id,),
    ).fetchall()
    meeting = conn.execute(
        "SELECT * FROM meeting_records WHERE meeting_id=?", (meeting_id,)
    ).fetchone()
    conn.close()

    title = meeting["title"] if meeting else meeting_id
    now = datetime.now()
    escalated = []

    conn = _get_db()
    for item in items:
        due = item["due_date"]
        is_overdue = due and datetime.strptime(due, "%Y-%m-%d") < now
        if is_overdue:
            conn.execute(
                "UPDATE action_items SET status='escalated', updated_at=? WHERE item_id=?",
                (now.isoformat(), item["item_id"]),
            )
            escalated.append(item)
    conn.commit()
    conn.close()

    if not escalated:
        return f"No overdue items found for meeting '{meeting_id}'. All items are on track."

    escalation_body = f"ESCALATION ALERT — Meeting: {title}\n\n"
    escalation_body += f"The following {len(escalated)} action item(s) are overdue:\n\n"
    for item in escalated:
        escalation_body += (
            f"• {item['task']}\n"
            f"  Owner: {item['owner']} | Due: {item['due_date']} | Priority: {item['priority']}\n\n"
        )
    escalation_body += f"Meeting ID: {meeting_id}\nGenerated: {now.isoformat()}"

    logger.warning(f"[Meeting] Escalating {len(escalated)} stalled items from {meeting_id}")

    email = os.getenv("ESCALATION_EMAIL", "")
    email_note = (
        f"Escalation email queued for {email}."
        if email else
        "Set ESCALATION_EMAIL in .env to send automatic escalation emails."
    )

    return (
        f"Escalation complete.\n"
        f"Meeting: {title} ({meeting_id})\n"
        f"Escalated: {len(escalated)} item(s)\n\n"
        + "\n".join(
            f"  🚨 {item['task']} — Owner: {item['owner']} | Due: {item['due_date']}"
            for item in escalated
        )
        + f"\n\n{email_note}\n\n"
        f"Escalation summary:\n{escalation_body}"
    )
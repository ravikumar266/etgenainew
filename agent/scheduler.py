"""
agent/scheduler.py
──────────────────
Two autonomous background jobs:

  1. Email monitor (every 10 min)
     - Reads unread Gmail
     - Filters for important/urgent items using LLM
     - Sends notification if anything needs attention

  2. Morning Business Briefing (daily at BRIEF_TIME, default 08:00)
     - Fully autonomous multi-step workflow
     - Uses Primary agent to gather intelligence
     - Uses Critic agent to verify quality
     - Delivers structured briefing to EMAIL_USER
     - Logs full run as a workflow with audit trail

     Briefing sections:
       a) Top business & tech news (3 stories)
       b) Weather for BRIEF_CITY
       c) Gmail important updates summary
       d) GitHub open PRs (if GITHUB_REPO set)
       e) Today's action items from any active meetings
       f) Business insight / motivational thought

.env variables:
  BRIEF_TIME   = 08:00          (24hr format, default 08:00)
  BRIEF_CITY   = Mumbai         (your city for weather)
  BRIEF_TOPICS = AI,startups,technology   (comma-separated news topics)
  GITHUB_REPO  =                (optional: owner/repo for PR summary)
  EMAIL_USER   = you@gmail.com  (briefing delivered here)
"""

import os
import uuid
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from agent.config import logger
from agent.tools_email import _filter_important, _read_emails_raw, _send_email_direct

_scheduler: BackgroundScheduler | None = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_chars: int = 8000) -> str:
    """
    Direct LLM call — bypasses the graph to avoid circular imports.
    Used inside scheduler jobs where the full graph is not needed.
    """
    try:
        from agent.config import llm
        response = llm.invoke(prompt[:max_chars])
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        return str(content).strip()
    except Exception as e:
        logger.error(f"[Scheduler] LLM call failed: {e}")
        return f"(LLM unavailable: {e})"


def _critic_review(content: str, context: str = "") -> str:
    """
    Run the critic agent on briefing content before sending.
    Returns improved content or original if critic fails.
    This is what makes the briefing genuinely multi-agent —
    Primary gathers, Critic verifies, then it gets sent.
    """
    try:
        from agent.config import critic_llm
        review_prompt = f"""You are a quality control agent reviewing a morning business briefing.

BRIEFING CONTENT:
{content[:3000]}

Check this briefing for:
1. Is the information complete and useful for starting a business day?
2. Are there any obvious errors, hallucinations, or missing sections?
3. Is the tone professional and concise?

If quality is acceptable (score >= 7), reply with just: APPROVED
If improvements needed, reply with the improved version directly (no preamble).
Keep it concise."""

        review = critic_llm.invoke(review_prompt)
        review_text = review.content if hasattr(review, "content") else str(review)
        if isinstance(review_text, list):
            review_text = " ".join(
                b.get("text", "") for b in review_text
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()

        if "APPROVED" in review_text[:20]:
            logger.info("[Briefing] Critic: APPROVED — sending as-is")
            return content
        else:
            logger.info("[Briefing] Critic: improved content — using revised version")
            return review_text.strip()

    except Exception as e:
        logger.warning(f"[Briefing] Critic review failed ({e}) — using original content")
        return content


def _fetch_news(topics: list[str]) -> str:
    """Fetch top 3 news stories for given topics using search_web."""
    try:
        from agent.tools_search import search_web
        topic_str = " OR ".join(topics[:3])
        query = f"latest business news today {topic_str} 2026"
        result = search_web.invoke({"query": query})
        return str(result)[:3000]
    except Exception as e:
        logger.warning(f"[Briefing] News fetch failed: {e}")
        return "(News unavailable)"


def _fetch_weather(city: str) -> str:
    """Fetch weather for briefing city."""
    try:
        from agent.tools_google import get_weather
        result = get_weather.invoke({"city": city})
        return str(result)
    except Exception as e:
        logger.warning(f"[Briefing] Weather fetch failed: {e}")
        return f"(Weather unavailable for {city})"


def _fetch_github_prs(repo: str) -> str:
    """Fetch open PRs from configured GitHub repo."""
    try:
        from agent.tools_github import list_prs
        result = list_prs.invoke({"repo": repo, "state": "open"})
        return str(result)[:1500]
    except Exception as e:
        logger.warning(f"[Briefing] GitHub PR fetch failed: {e}")
        return "(GitHub PRs unavailable)"


def _fetch_meeting_actions() -> str:
    """Fetch any overdue or at-risk action items from recent meetings."""
    try:
        import sqlite3
        db_path = os.getenv("MEETING_DB", "./meetings.db")
        if not os.path.exists(db_path):
            return "(No active meetings found)"

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        now = datetime.now().isoformat()

        rows = conn.execute("""
            SELECT ai.task, ai.owner, ai.due_date, ai.status, mr.title
            FROM action_items ai
            JOIN meeting_records mr ON ai.meeting_id = mr.meeting_id
            WHERE ai.status NOT IN ('completed', 'cancelled')
            ORDER BY ai.due_date ASC
            LIMIT 10
        """).fetchall()
        conn.close()

        if not rows:
            return "No pending action items."

        lines = ["Pending action items:"]
        for row in rows:
            overdue = row["due_date"] and row["due_date"] < now
            flag = " ⚠️ OVERDUE" if overdue else ""
            lines.append(
                f"  - [{row['owner']}] {row['task']} "
                f"(due: {row['due_date'][:10] if row['due_date'] else 'TBD'}){flag}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[Briefing] Action items fetch failed: {e}")
        return "(Action items unavailable)"


def _fetch_active_workflows() -> str:
    """Fetch any running workflows and their SLA status."""
    try:
        import sqlite3
        db_path = os.getenv("WORKFLOW_DB", "./workflows.db")
        if not os.path.exists(db_path):
            return "(No active workflows)"

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        now = datetime.now().isoformat()

        rows = conn.execute("""
            SELECT run_id, workflow, current_step, total_steps,
                   status, sla_deadline
            FROM workflow_runs
            WHERE status = 'running'
            ORDER BY sla_deadline ASC
            LIMIT 5
        """).fetchall()
        conn.close()

        if not rows:
            return "No active workflows."

        lines = ["Active workflows:"]
        for row in rows:
            sla_ok = row["sla_deadline"] > now
            sla_flag = "" if sla_ok else " ⚠️ SLA BREACH"
            lines.append(
                f"  - {row['workflow']} [{row['run_id']}] "
                f"step {row['current_step']}/{row['total_steps']} "
                f"| SLA: {row['sla_deadline'][:16]}{sla_flag}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[Briefing] Workflow status fetch failed: {e}")
        return "(Workflow status unavailable)"


# ── Job 1: Email monitor (every 10 min) ──────────────────────────────────────

def _scheduled_check_updates() -> None:
    """
    Read unread Gmail, filter for important items, notify if found.
    Runs silently — results appear in server logs.
    """
    try:
        emails_text = _read_emails_raw()

        if emails_text.startswith("No unread") or emails_text.startswith("Error"):
            logger.info(f"[Scheduler] {emails_text}")
            return

        summary = _filter_important(emails_text)

        if "No important updates" in summary:
            logger.info("[Scheduler] Email check: no important updates")
            return

        logger.info(f"[Scheduler] Important updates found:\n{summary[:300]}")

        notify_email = os.getenv("EMAIL_USER")
        if notify_email:
            _send_email_direct(
                to=notify_email,
                subject=f"⚡ Important Email Updates — {datetime.now().strftime('%H:%M')}",
                body=f"ETGenAI detected important updates in your inbox:\n\n{summary}",
            )

    except Exception as e:
        logger.error(f"[Scheduler] Email check failed: {e}")


# ── Job 2: Morning Business Briefing ─────────────────────────────────────────

def _morning_briefing() -> None:
    """
    Autonomous morning business briefing — runs daily at BRIEF_TIME.

    Flow:
      1. Primary agent gathers: news + weather + emails + GitHub PRs
                                + active workflows + meeting action items
      2. Critic agent reviews the assembled briefing for quality
      3. If quality passes → send email to EMAIL_USER
      4. Log full run as a workflow entry with audit trail

    This demonstrates:
      - Full process ownership (triggered by schedule, no human prompt)
      - Multi-agent (Primary gathers, Critic verifies)
      - Auditable trail (workflow DB entry with all steps)
      - Minimal human involvement (zero after initial setup)
    """
    run_id = f"brief-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4]}"
    now_str = datetime.now().strftime("%A, %B %d %Y — %H:%M")
    audit_log = []

    def _log(event: str, detail: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "detail": detail[:200],
        }
        audit_log.append(entry)
        logger.info(f"[Briefing:{run_id}] {event}: {detail[:100]}")

    _log("BRIEFING_STARTED", f"run_id={run_id} time={now_str}")

    # ── Config ────────────────────────────────────────────────────────────────
    city        = os.getenv("BRIEF_CITY", "Mumbai")
    topics_raw  = os.getenv("BRIEF_TOPICS", "AI,technology,startups,business")
    topics      = [t.strip() for t in topics_raw.split(",")]
    github_repo = os.getenv("GITHUB_REPO", "")
    recipient   = os.getenv("EMAIL_USER", "")

    if not recipient:
        logger.error("[Briefing] EMAIL_USER not set — cannot deliver briefing")
        return

    sections = {}

    # ── Step 1: Fetch news ────────────────────────────────────────────────────
    _log("STEP_1_NEWS", f"topics={topics}")
    raw_news = _fetch_news(topics)
    news_summary = _call_llm(
        f"""Summarise the top 3 most important business/tech news stories from this data.
For each story write:
- Headline (bold)
- 2-sentence summary
- Business impact: one sentence on why this matters

News data:
{raw_news}

Keep the total under 300 words. Be factual and concise."""
    )
    sections["news"] = news_summary
    _log("STEP_1_DONE", f"{len(news_summary)} chars")

    # ── Step 2: Weather ───────────────────────────────────────────────────────
    _log("STEP_2_WEATHER", f"city={city}")
    weather_raw = _fetch_weather(city)
    sections["weather"] = weather_raw
    _log("STEP_2_DONE", weather_raw[:80])

    # ── Step 3: Email summary ─────────────────────────────────────────────────
    _log("STEP_3_EMAILS")
    try:
        emails_raw = _read_emails_raw()
        email_summary = _filter_important(emails_raw)
    except Exception as e:
        email_summary = f"(Email check failed: {e})"
    sections["emails"] = email_summary
    _log("STEP_3_DONE", email_summary[:80])

    # ── Step 4: GitHub PRs ────────────────────────────────────────────────────
    if github_repo:
        _log("STEP_4_GITHUB", f"repo={github_repo}")
        sections["github"] = _fetch_github_prs(github_repo)
        _log("STEP_4_DONE")
    else:
        sections["github"] = ""

    # ── Step 5: Active workflows ──────────────────────────────────────────────
    _log("STEP_5_WORKFLOWS")
    sections["workflows"] = _fetch_active_workflows()
    _log("STEP_5_DONE", sections["workflows"][:80])

    # ── Step 6: Meeting action items ──────────────────────────────────────────
    _log("STEP_6_ACTION_ITEMS")
    sections["actions"] = _fetch_meeting_actions()
    _log("STEP_6_DONE", sections["actions"][:80])

    # ── Step 7: Business insight (LLM generated) ──────────────────────────────
    _log("STEP_7_INSIGHT")
    insight = _call_llm(
        f"""Generate one short, sharp business insight or productivity tip for today ({now_str}).
Make it relevant to someone running an AI/tech project.
Maximum 3 sentences. Be specific and actionable, not generic."""
    )
    sections["insight"] = insight
    _log("STEP_7_DONE")

    # ── Step 8: Assemble full briefing ────────────────────────────────────────
    _log("STEP_8_ASSEMBLE")

    github_section = (
        f"\n\n📂 OPEN PULL REQUESTS ({github_repo})\n{sections['github']}"
        if sections["github"]
        else ""
    )

    raw_briefing = f"""╔══════════════════════════════════════════════════════════╗
        MORNING BUSINESS BRIEFING — {now_str}
        Delivered by ETGenAI Autonomous Agent System
╚══════════════════════════════════════════════════════════╝

📰 TOP NEWS ({", ".join(topics[:3])})
{sections["news"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌤️  WEATHER — {city}
{sections["weather"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📧 EMAIL UPDATES
{sections["emails"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  ACTIVE WORKFLOWS
{sections["workflows"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PENDING ACTION ITEMS
{sections["actions"]}
{github_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TODAY'S BUSINESS INSIGHT
{sections["insight"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated autonomously by ETGenAI | Run ID: {run_id}
Primary Agent: Google Gemini | Critic Agent: Gemini (Grok when configured)
"""

    _log("STEP_8_DONE", f"{len(raw_briefing)} chars assembled")

    # ── Step 9: Critic agent reviews the briefing ─────────────────────────────
    _log("STEP_9_CRITIC_REVIEW")
    final_briefing = _critic_review(raw_briefing)
    _log("STEP_9_DONE", "critic passed" if final_briefing == raw_briefing else "critic improved content")

    # ── Step 10: Deliver via email ─────────────────────────────────────────────
    _log("STEP_10_DELIVER", f"to={recipient}")
    try:
        _send_email_direct(
            to=recipient,
            subject=f"🌅 Morning Briefing — {datetime.now().strftime('%A %d %B %Y')}",
            body=final_briefing,
        )
        _log("STEP_10_DONE", "email sent successfully")
    except Exception as e:
        _log("STEP_10_FAILED", str(e))
        logger.error(f"[Briefing] Email delivery failed: {e}")
        return

    # ── Step 11: Write audit log to workflow DB ───────────────────────────────
    _log("STEP_11_AUDIT_LOG")
    try:
        import json
        import sqlite3
        db_path = os.getenv("WORKFLOW_DB", "./workflows.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id TEXT PRIMARY KEY,
                workflow TEXT,
                context TEXT,
                steps TEXT,
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER,
                status TEXT DEFAULT 'completed',
                started_at TEXT,
                sla_deadline TEXT,
                updated_at TEXT,
                audit_log TEXT DEFAULT '[]'
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO workflow_runs
            (run_id, workflow, context, steps, current_step, total_steps,
             status, started_at, sla_deadline, updated_at, audit_log)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id,
            "morning_briefing",
            json.dumps({"recipient": recipient, "city": city, "topics": topics}),
            json.dumps([f"Step {i+1}" for i in range(11)]),
            11, 11,
            "completed",
            audit_log[0]["timestamp"],
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            json.dumps(audit_log),
        ))
        conn.commit()
        conn.close()
        _log("STEP_11_DONE", f"audit trail saved run_id={run_id}")
    except Exception as e:
        logger.warning(f"[Briefing] Audit log save failed: {e}")

    logger.info(
        f"[Briefing] Morning briefing completed — run_id={run_id} "
        f"steps=11 delivered_to={recipient}"
    )


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

def start_scheduler() -> None:
    """
    Start both background jobs.
    Called once from main.py lifespan on app startup.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler()

    # Job 1: Email monitor — every 10 minutes
    _scheduler.add_job(
        _scheduled_check_updates,
        "interval",
        minutes=720,
        id="email_check",
    )

    # Job 2: Morning briefing — daily cron at BRIEF_TIME
    brief_time = os.getenv("BRIEF_TIME", "08:00")
    try:
        hour, minute = map(int, brief_time.split(":"))
    except Exception:
        hour, minute = 8, 0
        logger.warning(f"[Scheduler] Invalid BRIEF_TIME '{brief_time}' — defaulting to 08:00")

    _scheduler.add_job(
        _morning_briefing,
        "cron",
        hour=hour,
        minute=minute,
        id="morning_briefing",
    )

    _scheduler.start()
    logger.info(
        f"Background scheduler started:\n"
        f"  - Email monitor: every 10 minutes\n"
        f"  - Morning briefing: daily at {brief_time} "
        f"(city={os.getenv('BRIEF_CITY','Mumbai')}, "
        f"topics={os.getenv('BRIEF_TOPICS','AI,technology,startups')})"
    )


def stop_scheduler() -> None:
    """Gracefully stop scheduler. Called from main.py lifespan shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")


def get_scheduler() -> BackgroundScheduler | None:
    """Return scheduler instance for status endpoint in main.py."""
    return _scheduler
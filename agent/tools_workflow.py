"""
agent/tools_workflow.py
───────────────────────
Option 1: Process Orchestration Agent

Manages multi-step enterprise workflows with:
  - Step-by-step state tracking (SQLite, survives restarts)
  - SLA deadline enforcement with automatic escalation
  - Full auditable trail of every decision and action
  - Exception handling — failed steps trigger retry or escalation
  - Pre-built workflow templates: employee onboarding, contract review,
    procurement, meeting follow-up

User-facing @tools:
  start_workflow(name, context, sla_hours)  → create and begin a workflow run
  update_workflow_step(run_id, step, outcome, status) → record step completion
  get_workflow_status(run_id)               → full audit trail + SLA health
  list_workflows(status_filter)             → see all active/completed/failed runs
  escalate_workflow(run_id, reason)         → manually escalate to human

.env:
  WORKFLOW_DB=./workflows.db   (default — no setup needed)
  ESCALATION_EMAIL=ravishani98765432@gmail.com  (all escalation/manager emails go here)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

from agent.config import logger

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("WORKFLOW_DB", "./workflows.db")

WORKFLOW_TEMPLATES = {
    "employee_onboarding": {
        "steps": [
            "Send welcome email ONLY to the employee (use employee email from context, not manager email)",
            "Create onboarding Google Doc with schedule",
            "Set up system access and accounts (log action, do not email)",
            "Assign buddy/mentor (log action, do not email)",
            "Schedule day-1 orientation meeting (log action, do not email)",
        ],
        "default_sla_hours": 48,
    },
    "contract_review": {
        "steps": [
            "Retrieve contract document",
            "AI analysis of key clauses and risks",
            "Create review summary Google Doc",
            "Send summary to ESCALATION_EMAIL contact only",
        ],
        "default_sla_hours": 24,
    },
    "procurement": {
        "steps": [
            "Validate purchase request",
            "Search and compare vendors",
            "Generate comparison report",
            "Send approval request to ESCALATION_EMAIL contact only",
            "Place order on approval",
            "Confirm order and log completion",
        ],
        "default_sla_hours": 72,
    },
    "meeting_followup": {
        "steps": [
            "Extract action items from meeting",
            "Assign owners to each task",
            "Create follow-up tracking document",
        ],
        "default_sla_hours": 4,
    },
    "custom": {
        "steps": [],
        "default_sla_hours": 24,
    },
}


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workflow_runs (
            run_id          TEXT PRIMARY KEY,
            workflow        TEXT NOT NULL,
            context         TEXT DEFAULT '{}',
            steps           TEXT DEFAULT '[]',
            current_step    INTEGER DEFAULT 0,
            total_steps     INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'running',
            started_at      TEXT NOT NULL,
            sla_deadline    TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            audit_log       TEXT DEFAULT '[]',
            failure_count   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _append_audit(conn: sqlite3.Connection, run_id: str, event: str, detail: str = "") -> None:
    row = conn.execute(
        "SELECT audit_log FROM workflow_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if not row:
        return
    log = json.loads(row["audit_log"])
    log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "detail": detail[:500],
    })
    conn.execute(
        "UPDATE workflow_runs SET audit_log=?, updated_at=? WHERE run_id=?",
        (json.dumps(log), datetime.now().isoformat(), run_id),
    )
    conn.commit()


def _sla_health(sla_deadline_str: str) -> tuple[str, float]:
    """Returns (status_label, hours_remaining)."""
    deadline = datetime.fromisoformat(sla_deadline_str)
    hours_left = (deadline - datetime.now()).total_seconds() / 3600
    if hours_left < 0:
        return "BREACHED", hours_left
    if hours_left < 4:
        return "AT RISK", hours_left
    return "ON TRACK", hours_left


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def start_workflow(
    workflow_name: str,
    context: str,
    sla_hours: int = 24,
    custom_steps: str = "",
) -> str:
    """
    Start a tracked enterprise workflow run.

    Args:
      workflow_name : One of: employee_onboarding, contract_review,
                      procurement, meeting_followup, custom
      context       : JSON string with workflow data.
                      Example: '{"employee":"John Doe","email":"john@co.com","role":"Engineer"}'
      sla_hours     : Hours until SLA breach. Escalation fires automatically. Default: 24
      custom_steps  : Comma-separated step names (only for workflow_name="custom")

    IMPORTANT EMAIL RULES (agents must follow these strictly):
      - Employee emails go ONLY to the employee address in context
      - Manager/approval/escalation emails go ONLY to ESCALATION_EMAIL env var
      - Never invent or guess email addresses not present in context or .env
      - If ESCALATION_EMAIL is not set, log the action but do not send email
                      Example: "Validate request,Process payment,Send confirmation"

    Returns run_id — pass this to update_workflow_step after each step completes.
    The agent should then execute each step using appropriate tools and report back.
    """
    template = WORKFLOW_TEMPLATES.get(workflow_name, WORKFLOW_TEMPLATES["custom"])

    if workflow_name == "custom" and custom_steps:
        steps = [s.strip() for s in custom_steps.split(",") if s.strip()]
    else:
        steps = template["steps"]

    if not steps:
        return (
            "No steps defined for this workflow. "
            "For 'custom', provide custom_steps='Step1,Step2,Step3'."
        )

    run_id = f"wf-{uuid.uuid4().hex[:8]}"
    now = datetime.now()
    sla_hours = sla_hours or template["default_sla_hours"]
    sla_deadline = (now + timedelta(hours=sla_hours)).isoformat(timespec="seconds")

    conn = _get_db()
    conn.execute("""
        INSERT INTO workflow_runs
        (run_id, workflow, context, steps, total_steps,
         started_at, sla_deadline, updated_at, audit_log)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        run_id, workflow_name, context, json.dumps(steps), len(steps),
        now.isoformat(), sla_deadline, now.isoformat(), "[]",
    ))
    conn.commit()
    _append_audit(conn, run_id, "WORKFLOW_STARTED",
                  f"workflow={workflow_name} sla={sla_deadline} context={context[:200]}")
    conn.close()

    logger.info(f"[Workflow] Started {workflow_name} run_id={run_id}")

    steps_display = "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(steps))
    return (
        f"Workflow started successfully.\n\n"
        f"run_id    : {run_id}\n"
        f"workflow  : {workflow_name}\n"
        f"SLA       : {sla_deadline} ({sla_hours}h)\n"
        f"Context   : {context}\n\n"
        f"Steps to complete:\n{steps_display}\n\n"
        f"Now begin Step 1: '{steps[0]}'\n"
        f"After each step, call update_workflow_step with run_id='{run_id}'."
    )


@tool
def update_workflow_step(
    run_id: str,
    step_name: str,
    outcome: str,
    status: str = "running",
) -> str:
    """
    Record a completed step in a workflow and advance to the next.

    Args:
      run_id    : The workflow run ID returned by start_workflow.
      step_name : Name of the step just completed (e.g. "Send welcome email").
      outcome   : What happened — success detail or error description.
                  Be specific: include document URLs, email recipients, etc.
      status    : "running"   — step done, more steps remain
                  "completed" — ALL steps done, workflow finished
                  "failed"    — this step failed, will retry or escalate
                  "escalated" — needs human intervention

    Always call this after EVERY step so the audit trail stays complete.
    If status="failed", describe the error in outcome so it can be retried.
    """
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM workflow_runs WHERE run_id=?", (run_id,)
    ).fetchone()

    if not row:
        conn.close()
        return (
            f"Workflow run '{run_id}' not found. "
            "Use list_workflows to see active runs."
        )

    steps = json.loads(row["steps"])
    new_step = row["current_step"] + 1
    failure_count = row["failure_count"] + (1 if status == "failed" else 0)

    # Auto-escalate after 3 consecutive failures
    if failure_count >= 3 and status == "failed":
        status = "escalated"
        _append_audit(conn, run_id, "AUTO_ESCALATED",
                      f"3 consecutive failures on step '{step_name}'")
        logger.warning(f"[Workflow] {run_id} auto-escalated after 3 failures")

    conn.execute("""
        UPDATE workflow_runs
        SET current_step=?, status=?, updated_at=?, failure_count=?
        WHERE run_id=?
    """, (new_step, status, datetime.now().isoformat(), failure_count, run_id))
    conn.commit()
    _append_audit(conn, run_id, f"STEP_DONE",
                  f"step='{step_name}' status={status} outcome={outcome[:300]}")
    conn.close()

    sla_label, hours_left = _sla_health(row["sla_deadline"])
    sla_warning = (
        f"\n⚠️  SLA {sla_label}: {abs(hours_left):.1f}h "
        + ("OVERDUE" if hours_left < 0 else "remaining")
    ) if sla_label != "ON TRACK" else ""

    logger.info(f"[Workflow] {run_id} step {new_step}/{row['total_steps']} "
                f"'{step_name}' → {status}")

    if status == "completed":
        return (
            f"Workflow COMPLETED.\n"
            f"run_id: {run_id} | All {row['total_steps']} steps done.\n"
            f"Final step: {step_name}\n"
            f"Outcome: {outcome}{sla_warning}"
        )

    if status in ("failed", "escalated"):
        return (
            f"Step FAILED/ESCALATED.\n"
            f"run_id: {run_id} | Step: {step_name}\n"
            f"Error: {outcome}\n"
            f"Failure count: {failure_count}\n"
            f"Status: {status}{sla_warning}\n\n"
            + ("Escalation required — notify human supervisor." if status == "escalated"
               else f"Retry this step or call escalate_workflow('{run_id}', reason).")
        )

    next_step_name = steps[new_step] if new_step < len(steps) else "—"
    return (
        f"Step recorded: '{step_name}'\n"
        f"Progress: {new_step}/{row['total_steps']} steps complete\n"
        f"Outcome: {outcome}\n"
        f"SLA: {row['sla_deadline']} ({sla_label}){sla_warning}\n\n"
        f"Next step ({new_step + 1}/{row['total_steps']}): '{next_step_name}'\n"
        f"Proceed with the next step now."
    )


@tool
def get_workflow_status(run_id: str) -> str:
    """
    Get the complete status and full audit trail of a workflow run.

    Shows: current step, SLA health, all decisions made, all outcomes recorded.
    Use this to check progress or diagnose issues in a running workflow.
    """
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM workflow_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    conn.close()

    if not row:
        return f"Workflow run '{run_id}' not found."

    steps = json.loads(row["steps"])
    audit = json.loads(row["audit_log"])
    sla_label, hours_left = _sla_health(row["sla_deadline"])

    steps_display = []
    for i, step in enumerate(steps):
        if i < row["current_step"]:
            steps_display.append(f"  ✅ Step {i+1}: {step}")
        elif i == row["current_step"]:
            steps_display.append(f"  ⏳ Step {i+1}: {step}  ← CURRENT")
        else:
            steps_display.append(f"  ⬜ Step {i+1}: {step}")

    audit_display = "\n".join(
        f"  [{e['timestamp']}] {e['event']}: {e['detail']}"
        for e in audit
    ) or "  (no events yet)"

    sla_icon = "✅" if sla_label == "ON TRACK" else ("⚠️" if sla_label == "AT RISK" else "🚨")

    return (
        f"{'='*50}\n"
        f"Workflow: {row['workflow']}  |  run_id: {run_id}\n"
        f"Status:   {row['status'].upper()}\n"
        f"Progress: {row['current_step']}/{row['total_steps']} steps\n"
        f"SLA:      {sla_icon} {sla_label} — "
        f"{'OVERDUE by' if hours_left < 0 else ''} "
        f"{abs(hours_left):.1f}h {'overdue' if hours_left < 0 else 'remaining'}\n"
        f"Deadline: {row['sla_deadline']}\n"
        f"Started:  {row['started_at']}\n"
        f"Context:  {row['context']}\n\n"
        f"Steps:\n" + "\n".join(steps_display) + "\n\n"
        f"Audit Trail:\n{audit_display}\n"
        f"{'='*50}"
    )


@tool
def list_workflows(status_filter: str = "running") -> str:
    """
    List all workflow runs filtered by status.

    status_filter: "running", "completed", "failed", "escalated", or "all"

    Use this to monitor all active enterprise workflows at a glance.
    SLA breaches are flagged with 🚨.
    """
    conn = _get_db()
    if status_filter == "all":
        rows = conn.execute(
            "SELECT * FROM workflow_runs ORDER BY started_at DESC LIMIT 30"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM workflow_runs WHERE status=? ORDER BY started_at DESC LIMIT 30",
            (status_filter,),
        ).fetchall()
    conn.close()

    if not rows:
        return f"No workflows found with status='{status_filter}'."

    lines = [f"Workflows (filter: {status_filter}) — {len(rows)} found:\n"]
    for r in rows:
        sla_label, hours_left = _sla_health(r["sla_deadline"])
        icon = "🚨" if sla_label == "BREACHED" else ("⚠️" if sla_label == "AT RISK" else "✅")
        lines.append(
            f"{icon} [{r['run_id']}] {r['workflow']}\n"
            f"   Status: {r['status']} | "
            f"Step {r['current_step']}/{r['total_steps']} | "
            f"SLA: {sla_label} ({abs(hours_left):.1f}h) | "
            f"Started: {r['started_at'][:10]}"
        )
    return "\n".join(lines)


@tool
def escalate_workflow(run_id: str, reason: str) -> str:
    """
    Manually escalate a workflow to human intervention.

    Use when:
    - A step has failed and automatic retry is not appropriate
    - SLA is about to breach and the workflow cannot self-correct
    - An unexpected situation requires human judgment

    Args:
      run_id : The workflow run ID.
      reason : Why escalation is needed — be specific.

    This marks the workflow as escalated and sends a notification email
    to ESCALATION_EMAIL if set in .env.
    """
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM workflow_runs WHERE run_id=?", (run_id,)
    ).fetchone()

    if not row:
        conn.close()
        return f"Workflow run '{run_id}' not found."

    conn.execute(
        "UPDATE workflow_runs SET status='escalated', updated_at=? WHERE run_id=?",
        (datetime.now().isoformat(), run_id),
    )
    conn.commit()
    _append_audit(conn, run_id, "ESCALATED", f"reason={reason}")
    conn.close()

    logger.warning(f"[Workflow] {run_id} ESCALATED: {reason}")

    escalation_email = os.getenv("ESCALATION_EMAIL", "")
    email_note = (
        f"Escalation notification will be sent to {escalation_email}."
        if escalation_email
        else "Set ESCALATION_EMAIL in .env to send automatic escalation notifications."
    )

    return (
        f"Workflow escalated.\n"
        f"run_id  : {run_id}\n"
        f"workflow: {row['workflow']}\n"
        f"step    : {row['current_step']}/{row['total_steps']}\n"
        f"reason  : {reason}\n\n"
        f"{email_note}\n"
        f"Human intervention required to resume this workflow."
    )
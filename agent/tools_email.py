"""
agent/tools_email.py
─────────────────────
Email tools:
  - _read_emails_raw()   private helper — fetches unread Gmail
  - _filter_important()  private helper — LLM triage
  - _send_email_direct() private helper — SMTP, no approval (scheduler only)
  - check_updates        @tool — user-facing: summarise important emails
  - send_email           @tool — user-facing: send email (requires HITL approval)
"""

import base64
import os
import smtplib
from email.mime.text import MIMEText

from langchain_core.tools import tool

from agent.config import gmail_service, llm, logger


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_emails_raw() -> str:
    """
    Fetch 5 most recent unread Gmail messages.
    Returns sender, subject, and decoded body preview for each.
    NOT a @tool — Gemini cannot call this directly.
    """
    try:
        service = gmail_service()
        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX", "UNREAD"],
            maxResults=5,
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return "No unread emails found."

        email_data = []
        for msg in messages:
            detail = service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="full",
            ).execute()

            headers = detail.get("payload", {}).get("headers", [])
            subject = next(
                (h["value"] for h in headers if h["name"] == "Subject"),
                "(no subject)",
            )
            sender = next(
                (h["value"] for h in headers if h["name"] == "From"),
                "(unknown sender)",
            )

            body_text = ""
            payload = detail.get("payload", {})
            parts = payload.get("parts", [payload])
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body_text = base64.urlsafe_b64decode(data).decode(
                            "utf-8", errors="ignore"
                        )
                        break

            preview = body_text[:500].strip() or detail.get("snippet", "")
            email_data.append(
                f"From: {sender}\nSubject: {subject}\nPreview: {preview}"
            )

        return "\n\n---\n\n".join(email_data)

    except Exception as e:
        return f"Error reading emails: {str(e)}"


def _filter_important(emails_text: str) -> str:
    """
    Ask the LLM to extract important items from raw email text.
    NOT a @tool — called only by check_updates and the scheduler.
    """
    prompt = f"""You are an email triage assistant.

Read the emails below and extract ONLY the important items:
- Deadlines or due dates
- Meeting requests or schedule changes
- Urgent or time-sensitive updates
- Job offers, interviews, or college admissions

Return clean bullet points. If nothing important is found, say "No important updates."

EMAILS:
{emails_text}
"""
    try:
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Failed to summarise emails: {str(e)}"


def _send_email_direct(to: str, subject: str, body: str) -> None:
    """
    Send email via SMTP with NO human approval gate.
    Private — used ONLY by the background scheduler for notifications.
    Never expose this as a @tool.
    """
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender or not password:
        logger.error("[Scheduler] EMAIL_USER or EMAIL_PASS not set")
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        logger.info(f"[Scheduler] Notification email sent to {to}")
    except Exception as e:
        logger.error(f"[Scheduler] Notification email failed: {e}")


# ── User-facing tools ─────────────────────────────────────────────────────────

@tool
def check_updates() -> str:
    """
    Read unread Gmail messages and return a bullet-point summary of important
    items: deadlines, meetings, urgent updates, job/college notifications.
    No input required.
    """
    emails_text = _read_emails_raw()

    if not emails_text or emails_text.startswith("No unread") or emails_text.startswith("Error"):
        return f"No updates to report: {emails_text}"

    summary = _filter_important(emails_text)
    return f"📢 Important Updates:\n\n{summary}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP. Requires human approval before executing."""
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    if not sender or not password:
        return "❌ EMAIL_USER or EMAIL_PASS not set in environment."
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        logger.info(f"Email sent to {to} | subject: '{subject}'")
        return f"✅ Email sent successfully to {to}"
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return f"❌ Email failed: {str(e)}"
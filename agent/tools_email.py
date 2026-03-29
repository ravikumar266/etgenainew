"""
agent/tools_email.py
─────────────────────
Email tools — Resend API (SMTP nahi, Render pe kaam karta hai!)

Setup:
  .env me ye add karo:
    RESEND_API_KEY = re_xxxxxxxxxxxx   ← Resend dashboard se
    EMAIL_USER     = you@gmail.com     ← Gmail read ke liye (OAuth)
    EMAIL_FROM     = you@yourdomain.com ← Resend verified email/domain

  Resend free tier: 3000 emails/month, 100/day — enough for personal use

Tools:
  check_updates    — Gmail unread emails padhna (OAuth, unchanged)
  send_email       — Email bhejna via Resend API (SMTP nahi!)

Private helpers:
  _read_emails_raw()    — Gmail fetch
  _filter_important()   — LLM triage
  _send_email_direct()  — Resend API, no approval (scheduler use karta hai)
"""

import base64
import os

import requests
from langchain_core.tools import tool

from agent.config import gmail_service, llm, logger


# ── Resend API helper ─────────────────────────────────────────────────────────

def _resend_send(to: str, subject: str, body: str) -> dict:
    """
    Resend API se email bhejo.
    SMTP nahi — pure HTTPS — Render pe perfectly kaam karta hai!

    Returns: { success: bool, message: str }
    """
    api_key  = os.getenv("RESEND_API_KEY", "")
    from_email = os.getenv("EMAIL_FROM", os.getenv("EMAIL_USER", ""))

    if not api_key:
        return {"success": False, "message": "RESEND_API_KEY not set in environment"}

    if not from_email:
        return {"success": False, "message": "EMAIL_FROM not set in environment"}

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "from":    from_email,
                "to":      [to],
                "subject": subject,
                "text":    body,
            },
            timeout=15,
        )

        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            logger.info(f"[Email] Resend success → {to} | id={data.get('id', '?')}")
            return {"success": True, "message": f"Email sent to {to}", "id": data.get("id")}

        else:
            error = response.json().get("message", response.text)
            logger.error(f"[Email] Resend failed: {response.status_code} — {error}")
            return {"success": False, "message": f"Resend error {response.status_code}: {error}"}

    except requests.exceptions.Timeout:
        return {"success": False, "message": "Resend API timeout — try again"}
    except Exception as e:
        return {"success": False, "message": f"Resend request failed: {str(e)}"}


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_emails_raw() -> str:
    """
    Gmail se 5 latest unread messages fetch karo.
    OAuth use karta hai — SMTP nahi.
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
            parts   = payload.get("parts", [payload])
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
    """LLM se important emails filter karo."""
    prompt = f"""You are an email triage assistant.

Read the emails below and extract ONLY the important items:
- Deadlines or due dates
- Meeting requests or schedule changes
- Urgent or time-sensitive updates
- Job offers, interviews, or college admissions

Return clean bullet points. If nothing important, say "No important updates."

EMAILS:
{emails_text}
"""
    try:
        result = llm.invoke(prompt)
        content = result.content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        return str(content).strip()
    except Exception as e:
        return f"Failed to summarise emails: {str(e)}"


def _send_email_direct(to: str, subject: str, body: str) -> None:
    """
    Resend API se email bhejo — NO human approval.
    Sirf scheduler use karta hai (morning briefing, notifications).
    Kabhi @tool mat banao isko.
    """
    result = _resend_send(to, subject, body)

    if result["success"]:
        logger.info(f"[Scheduler] Email sent to {to} via Resend ✓")
    else:
        logger.error(f"[Scheduler] Email failed: {result['message']}")


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
    """
    Send an email using Resend API.
    Requires human approval before executing (handled by graph).

    Args:
      to      : recipient email address
      subject : email subject line
      body    : plain text email body
    """
    result = _resend_send(to, subject, body)

    if result["success"]:
        logger.info(f"[Email] Sent to {to} | subject: '{subject}'")
        return f"✅ Email sent successfully to {to}"
    else:
        logger.error(f"[Email] Failed: {result['message']}")
        return f"❌ Email failed: {result['message']}"
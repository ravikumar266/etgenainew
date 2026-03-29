"""
agent/tools_email.py
─────────────────────
Email tools — Gmail API (primary) + Resend (fallback)

Priority:
  1. Gmail API (OAuth) — tumhara Gmail account use karta hai, koi domain verify nahi chahiye
  2. Resend API — agar Gmail fail ho toh fallback

Setup .env:
  EMAIL_USER     = you@gmail.com      ← Gmail read + send ke liye
  RESEND_API_KEY = re_xxxx            ← Optional fallback
  EMAIL_FROM     = you@yourdomain.com ← Resend ke liye (optional)
"""

import base64
import os
from email.mime.text import MIMEText

import requests
from langchain_core.tools import tool

from agent.config import gmail_service, llm, logger


# ── Gmail API send (Primary — koi domain verify nahi chahiye!) ─────────────────

def _gmail_api_send(to: str, subject: str, body: str) -> dict:
    """
    Gmail API se email bhejo — OAuth use karta hai.
    - SMTP nahi → Render pe kaam karta hai ✅
    - Domain verify nahi chahiye ✅
    - Kisi bhi Gmail/email pe bhej sakte ho ✅
    """
    try:
        service = gmail_service()

        msg            = MIMEText(body)
        msg["to"]      = to
        msg["subject"] = subject

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()

        logger.info(f"[Email] Gmail API → {to} ✓")
        return {"success": True, "message": f"Email sent to {to} via Gmail API"}

    except Exception as e:
        logger.warning(f"[Email] Gmail API failed: {e} — trying Resend")
        return {"success": False, "message": str(e)}


# ── Resend API send (Fallback) ────────────────────────────────────────────────

def _resend_send(to: str, subject: str, body: str) -> dict:
    """
    Resend API se email bhejo — SMTP nahi, pure HTTPS.
    Domain verify karna padega Resend me.
    Sirf fallback ke taur pe use hota hai.
    """
    api_key    = os.getenv("RESEND_API_KEY", "")
    from_email = os.getenv("EMAIL_FROM", os.getenv("EMAIL_USER", ""))

    if not api_key:
        return {"success": False, "message": "RESEND_API_KEY not set"}

    if not from_email:
        return {"success": False, "message": "EMAIL_FROM not set"}

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

        if response.status_code in (200, 201):
            data = response.json()
            logger.info(f"[Email] Resend → {to} ✓ id={data.get('id', '?')}")
            return {"success": True, "message": f"Email sent to {to} via Resend"}
        else:
            error = response.json().get("message", response.text)
            logger.error(f"[Email] Resend failed: {response.status_code} — {error}")
            return {"success": False, "message": f"Resend error: {error}"}

    except Exception as e:
        return {"success": False, "message": f"Resend request failed: {str(e)}"}


# ── Smart send — Gmail first, Resend fallback ─────────────────────────────────

def _send_email_smart(to: str, subject: str, body: str) -> dict:
    """
    Pehle Gmail API try karo, fail hone pe Resend try karo.
    """
    # Step 1: Gmail API
    result = _gmail_api_send(to, subject, body)
    if result["success"]:
        return result

    # Step 2: Resend fallback
    logger.info("[Email] Gmail failed — trying Resend fallback")
    result = _resend_send(to, subject, body)
    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_emails_raw() -> str:
    """Gmail se 5 latest unread messages fetch karo."""
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
                userId="me", id=msg["id"], format="full",
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
        result  = llm.invoke(prompt)
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
    Scheduler ke liye — no approval gate.
    Gmail API first, Resend fallback.
    """
    result = _send_email_smart(to, subject, body)
    if result["success"]:
        logger.info(f"[Scheduler] Email sent to {to} ✓")
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
    Send an email. Uses Gmail API (no domain needed) with Resend as fallback.
    Requires human approval before executing.

    Args:
      to      : recipient email address (any email — gmail, yahoo, etc.)
      subject : email subject line
      body    : plain text email body
    """
    result = _send_email_smart(to, subject, body)

    if result["success"]:
        logger.info(f"[Email] Sent to {to} | subject: '{subject}'")
        return f"✅ Email sent successfully to {to}"
    else:
        logger.error(f"[Email] Failed: {result['message']}")
        return f"❌ Email failed: {result['message']}"
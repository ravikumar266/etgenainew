"""
agent/tools_google.py
─────────────────────
Google Docs tools (merged) and weather.

OLD (3 tools, confusing):           NEW (2 tools, clear):
  create_google_doc(title)      →     google_doc(title, content)
  append_to_google_doc(id,text) →     update_google_doc(id, content, mode)
  read_google_doc(id)           →     (read is built into update_google_doc)

google_doc(title, content)
  → Creates a new doc AND writes content in one single call.
  → LLM never needs to chain create + append manually.

update_google_doc(document_id, content, mode)
  → mode="read"    → returns current doc content
  → mode="append"  → adds content to end of doc
  → mode="replace" → wipes doc and writes fresh content
"""

import os

import requests
from googleapiclient.errors import HttpError
from langchain_core.tools import tool

from agent.config import docs_service, logger


# ── Weather ───────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city using OpenWeatherMap.
    Returns temperature, feels-like, humidity, and condition.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "WEATHER_API_KEY is not set in environment."
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )
    try:
        data = requests.get(url, timeout=10).json()
        if data.get("cod") != 200:
            return f"Weather error: {data.get('message', 'unknown error')}"
        return (
            f"City: {city}\n"
            f"Temperature: {data['main']['temp']}°C\n"
            f"Feels like: {data['main']['feels_like']}°C\n"
            f"Humidity: {data['main']['humidity']}%\n"
            f"Condition: {data['weather'][0]['description']}"
        )
    except Exception as e:
        return f"Failed to fetch weather: {str(e)}"


# ── Google Docs (merged into 2 tools) ────────────────────────────────────────

@tool
def google_doc(title: str, content: str) -> str:
    """
    Create a new Google Doc with a title AND write content into it in one step.

    Use this whenever the user wants to:
      - "create a doc about X"
      - "save this to a Google Doc"
      - "write a report and put it in Docs"
      - "create a weather report doc"

    Args:
      title   : Title of the new document (e.g. "Weather Report - Brussels")
      content : Full text content to write into the document. Write the complete
                report/summary here — do not leave it blank or use a placeholder.

    Returns the document URL so the user can open it immediately.

    Example usage:
      google_doc(
        title="Brussels Weather Report",
        content="Weather Report for Brussels\\n\\nTemperature: 12°C\\nCondition: Cloudy..."
      )
    """
    if not content or not content.strip():
        return (
            "❌ content cannot be empty. "
            "Provide the full text you want written into the document."
        )

    try:
        svc = docs_service()

        # Step 1: Create blank doc
        doc = svc.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]
        logger.info(f"[Docs] Created '{title}' ({doc_id})")

        # Step 2: Write content immediately — single API call
        svc.documents().batchUpdate(
            documentId=doc_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": content.strip(),
                        }
                    }
                ]
            },
        ).execute()

        preview = content.strip()[:120] + ("..." if len(content) > 120 else "")
        logger.info(f"[Docs] Written {len(content)} chars to '{title}'")
        return (
            f"✅ Google Doc created and filled: '{title}'\n"
            f"URL: https://docs.google.com/document/d/{doc_id}/edit\n"
            f"Content preview: {preview}"
        )

    except HttpError as e:
        return f"❌ Error creating doc: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


@tool
def update_google_doc(document_id: str, content: str, mode: str = "append") -> str:
    """
    Read or update an existing Google Doc.

    Use this when the user:
      - Wants to read an existing doc           → mode="read"
      - Wants to add more content to a doc      → mode="append"
      - Wants to replace all content in a doc   → mode="replace"

    Args:
      document_id : The Google Doc ID extracted from its URL.
                    URL format: docs.google.com/document/d/<document_id>/edit
                    Example ID: 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
      content     : Text to write (ignored for mode="read")
      mode        : One of:
                      "read"    → return current document text (content ignored)
                      "append"  → add content to the end of the document
                      "replace" → delete all existing content and write fresh

    Returns confirmation with a preview of what was written, or the doc text for read.
    """
    mode = mode.lower().strip()
    if mode not in ("read", "append", "replace"):
        return f"❌ Invalid mode '{mode}'. Use 'read', 'append', or 'replace'."

    try:
        svc = docs_service()
        doc = svc.documents().get(documentId=document_id).execute()
        title = doc.get("title", "Untitled")

        # ── READ ──────────────────────────────────────────────────────────────
        if mode == "read":
            text_parts = []
            for block in doc.get("body", {}).get("content", []):
                para = block.get("paragraph")
                if para:
                    for el in para.get("elements", []):
                        tr = el.get("textRun")
                        if tr:
                            text_parts.append(tr.get("content", ""))
            text = "".join(text_parts).strip()
            logger.info(f"[Docs] Read '{title}' ({len(text)} chars)")
            return f"📄 Title: {title}\n\n{text or '(document is empty)'}"

        # ── APPEND ────────────────────────────────────────────────────────────
        if mode == "append":
            if not content or not content.strip():
                return "❌ content cannot be empty for mode='append'."
            end_index = doc["body"]["content"][-1]["endIndex"] - 1
            svc.documents().batchUpdate(
                documentId=document_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": end_index},
                                "text": f"\n{content.strip()}",
                            }
                        }
                    ]
                },
            ).execute()
            preview = content.strip()[:120] + ("..." if len(content) > 120 else "")
            logger.info(f"[Docs] Appended {len(content)} chars to '{title}'")
            return (
                f"✅ Appended to '{title}'\n"
                f"URL: https://docs.google.com/document/d/{document_id}/edit\n"
                f"Added: {preview}"
            )

        # ── REPLACE ───────────────────────────────────────────────────────────
        if mode == "replace":
            if not content or not content.strip():
                return "❌ content cannot be empty for mode='replace'."

            body_content = doc.get("body", {}).get("content", [])
            requests_list = []

            # Delete all existing content (if any beyond the initial newline)
            if len(body_content) > 1:
                end_index = body_content[-1]["endIndex"] - 1
                if end_index > 1:
                    requests_list.append({
                        "deleteContentRange": {
                            "range": {"startIndex": 1, "endIndex": end_index}
                        }
                    })

            # Insert fresh content
            requests_list.append({
                "insertText": {
                    "location": {"index": 1},
                    "text": content.strip(),
                }
            })

            svc.documents().batchUpdate(
                documentId=document_id,
                body={"requests": requests_list},
            ).execute()

            preview = content.strip()[:120] + ("..." if len(content) > 120 else "")
            logger.info(f"[Docs] Replaced content in '{title}' ({len(content)} chars)")
            return (
                f"✅ Replaced content in '{title}'\n"
                f"URL: https://docs.google.com/document/d/{document_id}/edit\n"
                f"New content: {preview}"
            )

    except HttpError as e:
        return f"❌ Google Docs API error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"
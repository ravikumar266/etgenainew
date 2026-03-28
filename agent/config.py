"""
agent/config.py
───────────────
Single source of truth for:
  - Environment / .env loading
  - LLM with multi-key Gemini rotation + Groq fallback + retry on 429
  - Google OAuth credentials + service builders
  - Tavily client
  - Cloud Run base URL

Model priority / key strategy:
  1. Rotate across ALL keys in GEMINI_API_KEYS on 429 (round-robin)
  2. If all Gemini keys are exhausted → fall back to Groq (GROQ_API_KEY)
  3. Groq model set via GROQ_MODEL (default: llama-3.3-70b-versatile)

.env variables used here:
  GEMINI_API_KEYS = key1,key2,key3   (comma-separated, at least one required)
  GEMINI_MODEL    = gemini-2.5-flash  (optional override, default gemini-2.0-flash)
  GROQ_API_KEY    = gsk_...           (optional — enables Groq fallback)
  GROQ_MODEL      = llama-3.3-70b-versatile  (optional, default above)
"""

import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Parse Gemini API keys ─────────────────────────────────────────────────────

def _parse_gemini_keys() -> list[str]:
    """
    Read GEMINI_API_KEYS (comma-separated) or fall back to GOOGLE_API_KEY.
    Deduplicates while preserving order.
    """
    raw = os.getenv("GEMINI_API_KEYS", "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    else:
        single = os.getenv("GOOGLE_API_KEY", "").strip()
        keys = [single] if single else []

    if not keys:
        logger.warning(
            "[LLM] No Gemini API keys found. "
            "Set GEMINI_API_KEYS=key1,key2,key3 in .env"
        )
        return [""]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = [k for k in keys if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]
    logger.info(f"[LLM] Loaded {len(unique)} Gemini API key(s)")
    return unique


_GEMINI_KEYS: list[str] = _parse_gemini_keys()

# Expose first key as GOOGLE_API_KEY for any code that still references it
_GOOGLE_API_KEY: str = _GEMINI_KEYS[0] if _GEMINI_KEYS else ""

# Gemini model name
_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Groq settings
_GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
_GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_retry_delay(error_str: str) -> int:
    """
    Parse retry delay seconds from a 429 error message.
    Google API returns: 'retryDelay': '26s' or 'retry in 26.38s'
    Returns delay in seconds, defaults to 15 if not found.
    """
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    if match:
        return min(int(float(match.group(1))) + 2, 60)
    return 15


def _is_quota_error(error: Exception) -> bool:
    s = str(error)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def _build_gemini(key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        google_api_key=key,
        temperature=0.3,
    )


def _build_groq():
    """Build a LangChain-compatible Groq LLM. Returns None if unavailable."""
    if not _GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq  # type: ignore
        groq = ChatGroq(
            model=_GROQ_MODEL,
            api_key=_GROQ_API_KEY,
            temperature=0.3,
        )
        logger.info(f"[LLM] Groq fallback ready: {_GROQ_MODEL}")
        return groq
    except ImportError:
        logger.warning(
            "[LLM] langchain-groq not installed — Groq fallback disabled. "
            "Install with: pip install langchain-groq"
        )
        return None
    except Exception as e:
        logger.warning(f"[LLM] Could not initialise Groq: {e}")
        return None


# ── Multi-key rotating LLM ────────────────────────────────────────────────────

class _RotatingLLM:
    """
    LLM wrapper with:
      - Round-robin rotation across all GEMINI_API_KEYS on 429
      - Short sleep extracted from error message before switching key
      - After all Gemini keys exhausted → Groq fallback (if configured)
      - Exposes .invoke() and .bind_tools() matching LangChain interface

    Key rotation strategy:
      For each invoke() call, on a 429 we try the next key immediately
      (after a brief sleep).  We do not give up on a key permanently;
      on the next top-level call the rotation continues from where it left off,
      giving all keys a chance to recover.
    """

    def __init__(self) -> None:
        self._key_index: int = 0
        self._bound_tools: list = []
        self._groq = _build_groq()
        self._llm = self._make_gemini(self._key_index)

        logger.info(
            f"[LLM] Initialised — model={_GEMINI_MODEL} "
            f"keys={len(_GEMINI_KEYS)} groq={'yes' if self._groq else 'no'}"
        )

    # ── Internal builders ─────────────────────────────────────────────────────

    def _make_gemini(self, key_idx: int) -> Any:
        """Build a Gemini LLM (with tools bound if any) for the given key index."""
        base = _build_gemini(_GEMINI_KEYS[key_idx])
        if self._bound_tools:
            return base.bind_tools(self._bound_tools)
        return base

    def _next_key(self) -> bool:
        """
        Advance to the next Gemini key.
        Returns True if a new key is available, False if all keys exhausted.
        """
        next_idx = (self._key_index + 1) % len(_GEMINI_KEYS)
        if next_idx == self._key_index:
            # Only one key — already tried it
            return False
        # Detect full rotation (back to start)
        if next_idx == 0 and self._key_index == len(_GEMINI_KEYS) - 1:
            logger.warning("[LLM] All Gemini keys tried in this rotation")
            self._key_index = next_idx
            return False
        self._key_index = next_idx
        logger.warning(
            f"[LLM] Rotating to Gemini key #{self._key_index + 1}/{len(_GEMINI_KEYS)}"
        )
        self._llm = self._make_gemini(self._key_index)
        return True

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """
        Invoke with automatic key rotation + Groq fallback.

        Per-call strategy:
          1. Try current Gemini key
          2. On 429: sleep briefly, rotate to next key, retry
          3. Repeat for all keys (max one full rotation)
          4. If still failing: try Groq
          5. If Groq also fails (or unavailable): raise
        """
        # Track which key we started on to detect full rotation
        start_index = self._key_index
        rotated_once = False

        while True:
            try:
                return self._llm.invoke(messages, **kwargs)

            except Exception as e:
                if not _is_quota_error(e):
                    raise  # Non-quota errors bubble up immediately

                delay = _extract_retry_delay(str(e))
                logger.warning(
                    f"[LLM] 429 on key #{self._key_index + 1} — "
                    f"sleeping {delay}s then rotating"
                )
                time.sleep(delay)

                advanced = self._next_key()

                if not advanced or (self._key_index == start_index and rotated_once):
                    # Full rotation completed — try Groq
                    break

                if self._key_index == start_index:
                    rotated_once = True  # allow one full pass

        # ── Groq fallback ─────────────────────────────────────────────────────
        if self._groq is not None:
            logger.warning("[LLM] All Gemini keys exhausted — using Groq fallback")
            try:
                groq_llm = self._groq
                if self._bound_tools:
                    groq_llm = groq_llm.bind_tools(self._bound_tools)
                return groq_llm.invoke(messages, **kwargs)
            except Exception as groq_err:
                logger.error(f"[LLM] Groq fallback also failed: {groq_err}")
                raise RuntimeError(
                    f"All Gemini keys and Groq fallback failed. "
                    f"Last Groq error: {groq_err}"
                ) from groq_err

        raise RuntimeError(
            "All Gemini API keys are rate-limited and no Groq fallback is configured. "
            "Set GROQ_API_KEY in .env or wait for Gemini quota to reset (midnight PT)."
        )

    def bind_tools(self, tools: list) -> "_RotatingLLM":
        """
        Bind tools to the LLM — mirrors LangChain interface.
        Returns self so callers can do: llm_with_tools = llm.bind_tools(TOOLS)
        """
        self._bound_tools = list(tools)
        self._llm = self._make_gemini(self._key_index)
        return self

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute access to the underlying LLM."""
        return getattr(self._llm, name)


# ── Singleton LLM ─────────────────────────────────────────────────────────────

llm = _RotatingLLM()

# Critic LLM — same pool, independent instance so it rotates separately
critic_llm = _RotatingLLM()

logger.info(
    f"[LLM] Config summary:\n"
    f"      Gemini model : {_GEMINI_MODEL}\n"
    f"      Gemini keys  : {len(_GEMINI_KEYS)} key(s) — rotating on 429\n"
    f"      Groq fallback: {_GROQ_MODEL if _GROQ_API_KEY else 'disabled (no GROQ_API_KEY)'}\n"
    f"      Override model via GEMINI_MODEL in .env"
)


# ── Google OAuth ──────────────────────────────────────────────────────────────

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_creds() -> Credentials:
    """Load saved OAuth token or open browser login on first run."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
        logger.info("token.json saved ✓")
    return creds


def docs_service():
    return build("docs", "v1", credentials=get_creds())


def gmail_service():
    return build("gmail", "v1", credentials=get_creds())


# ── Tavily client ─────────────────────────────────────────────────────────────

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ── Cloud Run URL ─────────────────────────────────────────────────────────────

CLOUD_RUN_URL = os.getenv("CLOUD_RUN_URL", "")
"""
agent/tools_rag.py
──────────────────
RAG (Retrieval-Augmented Generation) tools.

Ingestion sources:
  1. Web pages    — scrapes any URL, chunks text, embeds into ChromaDB
  2. PDF files    — extracts text page-by-page, chunks, embeds into ChromaDB
  3. YouTube      — fetches transcript + metadata via YouTube Data API v3
                    and youtube-transcript-api, chunks, embeds into ChromaDB

Storage:
  - ChromaDB (persisted to ./chroma_db/ — survives restarts)
  - Google text-embedding-004 (preferred) OR HuggingFace all-MiniLM-L6-v2 (local fallback)
    Uses the same GOOGLE_API_KEY as the LLM — no extra key needed.

User-facing @tools:
  ingest_webpage(url, collection)          → webpage → chunks → store
  ingest_pdf(file_path, collection)        → PDF     → chunks → store
  ingest_youtube(video_url, collection)    → YouTube → transcript + metadata → store
  query_rag(question, collection, top_k)   → similarity search → return chunks
  list_rag_collections()                   → list all collections + sizes
  delete_rag_collection(collection)        → permanently wipe a collection

Required .env:
  GOOGLE_API_KEY    — for LLM + embeddings (already set)
  YOUTUBE_API_KEY   — YouTube Data API v3 key (new — see setup below)

Required packages (pip install):
  chromadb
  langchain-community
  langchain-google-genai
  pypdf
  beautifulsoup4
  requests
  youtube-transcript-api
  google-api-python-client      ← for YouTube Data API v3

Setup for YOUTUBE_API_KEY:
  1. Go to https://console.cloud.google.com/
  2. Enable "YouTube Data API v3"
  3. Create an API key under APIs & Services → Credentials
  4. Add to .env:  YOUTUBE_API_KEY=your_key_here
"""

import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from agent.config import logger

# ── Embeddings ────────────────────────────────────────────────────────────────
# Priority:
#   1. Google text-embedding-004  (needs GOOGLE_API_KEY, best quality)
#   2. HuggingFace all-MiniLM-L6-v2 (free, runs locally, no API key)
#
# Force a provider by setting in .env:
#   EMBEDDING_PROVIDER=google        or
#   EMBEDDING_PROVIDER=huggingface
# Default is "auto" — tries Google first, falls back to HuggingFace.

def _build_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()

    if provider in ("auto", "google"):
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                emb = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=api_key,
                )
                emb.embed_query("test")
                logger.info("[RAG] Embeddings: Google text-embedding-004")
                return emb
            except Exception as e:
                logger.warning(f"[RAG] Google embeddings failed ({e}) — trying HuggingFace")

    if provider in ("auto", "huggingface"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                raise RuntimeError(
                    "Install HuggingFace embeddings: "
                    "pip install langchain-huggingface sentence-transformers"
                )
        try:
            emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("[RAG] Embeddings: HuggingFace all-MiniLM-L6-v2 (local)")
            return emb
        except Exception as e:
            raise RuntimeError(f"HuggingFace embeddings failed: {e}")

    raise RuntimeError(
        "No embedding provider available. Set GOOGLE_API_KEY or install: "
        "pip install langchain-huggingface sentence-transformers"
    )

_embeddings = _build_embeddings()

# ── Chroma persist directory ──────────────────────────────────────────────────

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# ── Text splitter (shared across all sources) ─────────────────────────────────
# 1000-char chunks, 150-char overlap to preserve context across boundaries.

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_vectorstore(collection: str) -> Chroma:
    """Load or create a named Chroma collection."""
    return Chroma(
        collection_name=_safe_collection_name(collection),
        embedding_function=_embeddings,
        persist_directory=CHROMA_DIR,
    )


def _safe_collection_name(name: str) -> str:
    """
    Chroma collection names: 3-63 chars, alphanumeric + hyphens only.
    """
    safe = re.sub(r"[^a-z0-9\-]", "-", name.lower().strip())
    safe = re.sub(r"-+", "-", safe).strip("-")
    safe = safe[:63]
    if len(safe) < 3:
        safe = (safe + "---")[:3]
    return safe


def _ingest_text(
    text: str,
    collection: str,
    source_label: str,
    metadata_extra: Optional[dict] = None,
) -> str:
    """Core pipeline: chunk → embed → upsert. Returns user-facing summary."""
    chunks = _splitter.split_text(text)
    if not chunks:
        return f"No text could be extracted from '{source_label}'."

    metadata = [
        {"source": source_label, "chunk": i, **(metadata_extra or {})}
        for i in range(len(chunks))
    ]

    vs = _get_vectorstore(collection)
    vs.add_texts(texts=chunks, metadatas=metadata)

    logger.info(
        f"[RAG] Ingested '{source_label}' → collection='{collection}' "
        f"({len(chunks)} chunks)"
    )
    return (
        f"✅ Ingested {len(chunks)} chunks from '{source_label}' "
        f"into collection '{collection}'."
    )


def _scrape_url(url: str) -> str:
    """Fetch a webpage, strip boilerplate, return plain text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_pdf(file_path: str) -> str:
    """Extract text page-by-page from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def _extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from any URL format:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://youtu.be/VIDEO_ID
      https://youtube.com/shorts/VIDEO_ID
      https://www.youtube.com/embed/VIDEO_ID
    Returns None if the URL is not a recognised YouTube URL.
    """
    parsed = urlparse(url)

    # youtu.be/VIDEO_ID
    if parsed.netloc in ("youtu.be",):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if vid else None

    # youtube.com/watch?v=VIDEO_ID
    if parsed.netloc in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            ids = qs.get("v", [])
            return ids[0] if ids else None
        # /shorts/VIDEO_ID  or  /embed/VIDEO_ID  or  /v/VIDEO_ID
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
            return parts[1]

    return None


def _fetch_youtube_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a YouTube video using youtube-transcript-api.
    Tries English first, then falls back to any available language.
    Returns plain text with timestamps stripped.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        raise ImportError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        )

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created English, then auto-generated English, then any language
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            pass

        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            except NoTranscriptFound:
                pass

        if not transcript:
            # Fall back to first available language and translate to English
            for t in transcript_list:
                transcript = t
                if t.language_code != "en":
                    logger.info(f"[RAG] Translating transcript from '{t.language_code}' to English")
                    transcript = t.translate("en")
                break

        if not transcript:
            return ""

        entries = transcript.fetch()
        # Join all text pieces, stripping timestamps
        lines = [entry["text"].strip() for entry in entries if entry.get("text", "").strip()]
        return " ".join(lines)

    except TranscriptsDisabled:
        return ""
    except Exception as e:
        logger.warning(f"[RAG] Transcript fetch failed for {video_id}: {e}")
        return ""


def _fetch_youtube_metadata(video_id: str) -> dict:
    """
    Fetch video title, description, channel, publish date, and tags
    using the YouTube Data API v3.

    Returns a dict — all fields default to empty string on failure so
    the caller never needs to guard against KeyError.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    result = {
        "title": "",
        "description": "",
        "channel": "",
        "published_at": "",
        "tags": [],
        "duration": "",
        "view_count": "",
    }

    if not api_key:
        logger.warning("[RAG] YOUTUBE_API_KEY not set — skipping metadata fetch")
        return result

    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails,statistics"
        f"&id={video_id}&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            logger.warning(f"[RAG] No YouTube metadata returned for video_id={video_id}")
            return result

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})

        result["title"] = snippet.get("title", "")
        result["description"] = snippet.get("description", "")
        result["channel"] = snippet.get("channelTitle", "")
        result["published_at"] = snippet.get("publishedAt", "")
        result["tags"] = snippet.get("tags", [])
        result["duration"] = details.get("duration", "")         # ISO 8601 e.g. PT4M13S
        result["view_count"] = stats.get("viewCount", "")

    except Exception as e:
        logger.warning(f"[RAG] YouTube metadata fetch failed: {e}")

    return result


def _build_youtube_text(video_id: str, metadata: dict, transcript: str) -> str:
    """
    Assemble a single document string from YouTube metadata + transcript.
    Structure is designed so the splitter keeps meaningful context together.
    """
    lines = []

    if metadata.get("title"):
        lines.append(f"Title: {metadata['title']}")
    if metadata.get("channel"):
        lines.append(f"Channel: {metadata['channel']}")
    if metadata.get("published_at"):
        lines.append(f"Published: {metadata['published_at'][:10]}")
    if metadata.get("view_count"):
        lines.append(f"Views: {metadata['view_count']}")
    if metadata.get("duration"):
        lines.append(f"Duration: {metadata['duration']}")
    if metadata.get("tags"):
        lines.append(f"Tags: {', '.join(metadata['tags'][:10])}")

    lines.append(f"\nVideo URL: https://www.youtube.com/watch?v={video_id}\n")

    if metadata.get("description"):
        desc = metadata["description"][:2000]   # cap description at 2000 chars
        lines.append(f"Description:\n{desc}\n")

    if transcript:
        lines.append(f"Transcript:\n{transcript}")
    else:
        lines.append("Transcript: (not available — video may have transcripts disabled)")

    return "\n".join(lines)


# ── User-facing @tools ────────────────────────────────────────────────────────

@tool
def ingest_webpage(url: str, collection: str = "default") -> str:
    """
    Load a webpage, split it into chunks, and store in the RAG vector database.

    Args:
      url        : Full URL of the webpage (e.g. https://example.com/article)
      collection : Knowledge base name to store into (default: "default").
                   Use different names to keep topics separate, e.g. "ai-news", "docs".

    Always call this before query_rag when answering questions about a specific webpage.
    """
    try:
        text = _scrape_url(url)
        if not text:
            return f"The page at {url} appears to be empty or unreadable."
        return _ingest_text(
            text=text,
            collection=collection,
            source_label=url,
            metadata_extra={"type": "webpage"},
        )
    except requests.exceptions.ConnectionError:
        return f"Could not connect to {url}. Check the URL and your internet connection."
    except requests.exceptions.Timeout:
        return f"Request to {url} timed out after 15 seconds."
    except Exception as e:
        return f"Failed to ingest webpage: {str(e)}"


@tool
def ingest_pdf(file_path: str, collection: str = "default") -> str:
    """
    Load a PDF file, split it into chunks, and store in the RAG vector database.

    Args:
      file_path  : Absolute or relative path to the PDF on disk.
                   Windows example: "C:/Users/You/Documents/report.pdf"
      collection : Knowledge base name to store into (default: "default").

    Always call this before query_rag when answering questions about a PDF.
    """
    if not os.path.exists(file_path):
        return (
            f"File not found: '{file_path}'. "
            "Please provide the full absolute path to the PDF."
        )
    if not file_path.lower().endswith(".pdf"):
        return f"'{file_path}' does not appear to be a PDF file."

    try:
        text = _load_pdf(file_path)
        if not text:
            return (
                f"No readable text found in '{file_path}'. "
                "The PDF may be scanned/image-based and requires OCR."
            )
        return _ingest_text(
            text=text,
            collection=collection,
            source_label=os.path.basename(file_path),
            metadata_extra={"type": "pdf", "path": file_path},
        )
    except Exception as e:
        return f"Failed to ingest PDF: {str(e)}"


@tool
def ingest_youtube(video_url: str, collection: str = "default") -> str:
    """
    Load a YouTube video's transcript and metadata into the RAG vector database.

    What gets ingested:
      - Full transcript (auto-generated or manual captions)
      - Video title, description, channel, publish date, tags, view count
      - Metadata is prepended to the transcript for context

    Args:
      video_url  : Any valid YouTube URL format, e.g.
                   https://www.youtube.com/watch?v=dQw4w9WgXcQ
                   https://youtu.be/dQw4w9WgXcQ
                   https://youtube.com/shorts/VIDEO_ID
      collection : Knowledge base name to store into (default: "default").

    Notes:
      - Requires YOUTUBE_API_KEY in .env for metadata (title, description, etc.)
      - Transcript works WITHOUT an API key for most public videos
      - If the video has disabled captions, only metadata will be ingested
      - Always call query_rag after this to answer questions about the video
    """
    # Step 1: Extract video ID
    video_id = _extract_video_id(video_url)
    if not video_id:
        return (
            f"Could not extract a YouTube video ID from '{video_url}'. "
            "Please provide a valid YouTube URL."
        )

    logger.info(f"[RAG] Ingesting YouTube video: {video_id}")

    # Step 2: Fetch metadata (requires YOUTUBE_API_KEY)
    metadata = _fetch_youtube_metadata(video_id)
    title = metadata.get("title") or f"YouTube video {video_id}"

    # Step 3: Fetch transcript (no API key required for public videos)
    transcript = _fetch_youtube_transcript(video_id)
    if not transcript:
        logger.warning(f"[RAG] No transcript found for video {video_id}")

    # Step 4: Assemble full text document
    full_text = _build_youtube_text(video_id, metadata, transcript)

    if not full_text.strip():
        return (
            f"Could not retrieve any content for video '{video_url}'. "
            "The video may be private, age-restricted, or have no transcript."
        )

    # Step 5: Chunk → embed → store
    source_label = f"youtube:{video_id}"
    result = _ingest_text(
        text=full_text,
        collection=collection,
        source_label=source_label,
        metadata_extra={
            "type": "youtube",
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "channel": metadata.get("channel", ""),
            "has_transcript": bool(transcript),
        },
    )

    # Append helpful context to the return message
    transcript_status = (
        f"Transcript: {len(transcript.split())} words ingested."
        if transcript
        else "Transcript: not available (captions may be disabled)."
    )
    return f"{result}\nTitle: {title}\n{transcript_status}"


@tool
def query_rag(question: str, collection: str = "default", top_k: int = 4) -> str:
    """
    Search the RAG vector database and return the most relevant text chunks.

    Args:
      question   : The question or search query.
      collection : Which knowledge base to search (default: "default").
                   Must match the collection name used during ingest.
      top_k      : Number of chunks to retrieve (default: 4, max: 10).

    Always synthesize the retrieved chunks into a clear answer.
    Cite the source (URL, filename, or YouTube video) for each key point.
    Never dump raw chunks directly at the user.
    """
    try:
        vs = _get_vectorstore(collection)

        count = vs._collection.count()
        if count == 0:
            return (
                f"The collection '{collection}' is empty. "
                "Use ingest_webpage, ingest_pdf, or ingest_youtube to add content first."
            )

        top_k = min(max(1, top_k), 10)
        results = vs.similarity_search_with_relevance_scores(question, k=top_k)

        if not results:
            return f"No relevant results found in collection '{collection}' for: {question}"

        output_parts = [f"Retrieved {len(results)} chunks from '{collection}':\n"]

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("type", "unknown").upper()
            chunk_num = doc.metadata.get("chunk", "?")
            relevance = f"{score:.0%}"

            # Show richer label for YouTube results
            if doc_type == "YOUTUBE":
                yt_title = doc.metadata.get("title", "")
                yt_url = doc.metadata.get("video_url", source)
                label = f"[{i}] YOUTUBE | '{yt_title}' | {yt_url} | chunk {chunk_num} | relevance {relevance}"
            else:
                label = f"[{i}] {doc_type} | {source} | chunk {chunk_num} | relevance {relevance}"

            output_parts.append(f"{label}\n{doc.page_content.strip()}\n")

        logger.info(
            f"[RAG] query='{question[:60]}' collection='{collection}' "
            f"returned {len(results)} chunks"
        )
        return "\n".join(output_parts)

    except Exception as e:
        return f"RAG query failed: {str(e)}"


@tool
def list_rag_collections() -> str:
    """
    List all available RAG knowledge base collections and their chunk counts.
    Use this to see what content has been ingested before running query_rag.
    No input required.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()

        if not collections:
            return (
                "No RAG collections found. "
                "Use ingest_webpage, ingest_pdf, or ingest_youtube to create one."
            )

        lines = [f"Found {len(collections)} RAG collection(s):\n"]
        for col in collections:
            count = col.count()
            lines.append(f"  - '{col.name}' → {count} chunks stored")

        return "\n".join(lines)

    except Exception as e:
        return f"Failed to list collections: {str(e)}"


@tool
def delete_rag_collection(collection: str) -> str:
    """
    Permanently delete a RAG collection and all its stored chunks.
    Use this to remove outdated or unwanted ingested content.

    Args:
      collection : Name of the collection to delete.
    """
    try:
        import chromadb
        safe_name = _safe_collection_name(collection)
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        existing = [c.name for c in client.list_collections()]
        if safe_name not in existing:
            return (
                f"Collection '{collection}' not found. "
                f"Available: {', '.join(existing) or 'none'}"
            )

        client.delete_collection(safe_name)
        logger.info(f"[RAG] Deleted collection '{safe_name}'")
        return f"Collection '{collection}' deleted successfully."

    except Exception as e:
        return f"Failed to delete collection: {str(e)}"
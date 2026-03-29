"""
agent/tools_rag.py
──────────────────
RAG (Retrieval-Augmented Generation) tools.
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

# ── Embeddings — LAZY LOADING (startup pe crash nahi hoga) ───────────────────

_embeddings = None  # startup pe None — pehli use pe load hoga

def _build_embeddings():
    """Google embeddings try karo — fail hone pe error do."""
    api_key = os.getenv("GOOGLE_API_KEY", "")

    if api_key:
        # Try multiple Google embedding models
        models_to_try = [
            "models/text-embedding-004",
            "models/embedding-001",
            "models/gemini-embedding-exp-03-07",
        ]
        for model_name in models_to_try:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                emb = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=api_key,
                )
                emb.embed_query("test")
                logger.info(f"[RAG] Embeddings: Google {model_name}")
                return emb
            except Exception as e:
                logger.warning(f"[RAG] Model {model_name} failed: {e}")
                continue

    # Fallback — ChromaDB ka built-in embedding use karo (no torch needed)
    logger.warning("[RAG] Google embeddings unavailable — using ChromaDB default embeddings")
    return None  # None matlab Chroma apna default use karega


def _get_embeddings():
    """Lazy load embeddings — pehli baar call hone pe build karta hai."""
    global _embeddings
    if _embeddings is None:
        _embeddings = _build_embeddings()
    return _embeddings


# ── Chroma persist directory ──────────────────────────────────────────────────

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# ── Text splitter ─────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_vectorstore(collection: str) -> Chroma:
    """Load or create a named Chroma collection."""
    emb = _get_embeddings()
    kwargs = dict(
        collection_name=_safe_collection_name(collection),
        persist_directory=CHROMA_DIR,
    )
    if emb is not None:
        kwargs["embedding_function"] = emb
    return Chroma(**kwargs)


def _safe_collection_name(name: str) -> str:
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
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_pdf(file_path: str) -> str:
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
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be",):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if vid else None
    if parsed.netloc in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            ids = qs.get("v", [])
            return ids[0] if ids else None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
            return parts[1]
    return None


def _fetch_youtube_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        raise ImportError("youtube-transcript-api is required")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
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
            for t in transcript_list:
                transcript = t
                if t.language_code != "en":
                    transcript = t.translate("en")
                break
        if not transcript:
            return ""
        entries = transcript.fetch()
        lines = [entry["text"].strip() for entry in entries if entry.get("text", "").strip()]
        return " ".join(lines)
    except TranscriptsDisabled:
        return ""
    except Exception as e:
        logger.warning(f"[RAG] Transcript fetch failed for {video_id}: {e}")
        return ""


def _fetch_youtube_metadata(video_id: str) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    result = {
        "title": "", "description": "", "channel": "",
        "published_at": "", "tags": [], "duration": "", "view_count": "",
    }
    if not api_key:
        return result
    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
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
        result["duration"] = details.get("duration", "")
        result["view_count"] = stats.get("viewCount", "")
    except Exception as e:
        logger.warning(f"[RAG] YouTube metadata fetch failed: {e}")
    return result


def _build_youtube_text(video_id: str, metadata: dict, transcript: str) -> str:
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
        lines.append(f"Description:\n{metadata['description'][:2000]}\n")
    if transcript:
        lines.append(f"Transcript:\n{transcript}")
    else:
        lines.append("Transcript: (not available)")
    return "\n".join(lines)


# ── User-facing @tools ────────────────────────────────────────────────────────

@tool
def ingest_webpage(url: str, collection: str = "default") -> str:
    """Load a webpage and store in RAG vector database."""
    try:
        text = _scrape_url(url)
        if not text:
            return f"The page at {url} appears to be empty or unreadable."
        return _ingest_text(text=text, collection=collection, source_label=url,
                            metadata_extra={"type": "webpage"})
    except requests.exceptions.ConnectionError:
        return f"Could not connect to {url}."
    except requests.exceptions.Timeout:
        return f"Request to {url} timed out."
    except Exception as e:
        return f"Failed to ingest webpage: {str(e)}"


@tool
def ingest_pdf(file_path: str, collection: str = "default") -> str:
    """Load a PDF file and store in RAG vector database."""
    if not os.path.exists(file_path):
        return f"File not found: '{file_path}'."
    if not file_path.lower().endswith(".pdf"):
        return f"'{file_path}' does not appear to be a PDF file."
    try:
        text = _load_pdf(file_path)
        if not text:
            return f"No readable text found in '{file_path}'."
        return _ingest_text(text=text, collection=collection,
                            source_label=os.path.basename(file_path),
                            metadata_extra={"type": "pdf", "path": file_path})
    except Exception as e:
        return f"Failed to ingest PDF: {str(e)}"


@tool
def ingest_youtube(video_url: str, collection: str = "default") -> str:
    """Load a YouTube video transcript and metadata into RAG vector database."""
    video_id = _extract_video_id(video_url)
    if not video_id:
        return f"Could not extract YouTube video ID from '{video_url}'."
    logger.info(f"[RAG] Ingesting YouTube video: {video_id}")
    metadata = _fetch_youtube_metadata(video_id)
    title = metadata.get("title") or f"YouTube video {video_id}"
    transcript = _fetch_youtube_transcript(video_id)
    full_text = _build_youtube_text(video_id, metadata, transcript)
    if not full_text.strip():
        return f"Could not retrieve any content for video '{video_url}'."
    result = _ingest_text(
        text=full_text, collection=collection,
        source_label=f"youtube:{video_id}",
        metadata_extra={
            "type": "youtube", "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title, "channel": metadata.get("channel", ""),
            "has_transcript": bool(transcript),
        },
    )
    transcript_status = (
        f"Transcript: {len(transcript.split())} words ingested."
        if transcript else "Transcript: not available."
    )
    return f"{result}\nTitle: {title}\n{transcript_status}"


@tool
def query_rag(question: str, collection: str = "default", top_k: int = 4) -> str:
    """Search the RAG vector database and return relevant chunks."""
    try:
        vs = _get_vectorstore(collection)
        count = vs._collection.count()
        if count == 0:
            return (f"The collection '{collection}' is empty. "
                    "Use ingest_webpage, ingest_pdf, or ingest_youtube to add content first.")
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
            if doc_type == "YOUTUBE":
                yt_title = doc.metadata.get("title", "")
                yt_url = doc.metadata.get("video_url", source)
                label = f"[{i}] YOUTUBE | '{yt_title}' | {yt_url} | chunk {chunk_num} | relevance {relevance}"
            else:
                label = f"[{i}] {doc_type} | {source} | chunk {chunk_num} | relevance {relevance}"
            output_parts.append(f"{label}\n{doc.page_content.strip()}\n")
        logger.info(f"[RAG] query='{question[:60]}' collection='{collection}' returned {len(results)} chunks")
        return "\n".join(output_parts)
    except Exception as e:
        return f"RAG query failed: {str(e)}"


@tool
def list_rag_collections() -> str:
    """List all available RAG knowledge base collections."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        if not collections:
            return "No RAG collections found."
        lines = [f"Found {len(collections)} RAG collection(s):\n"]
        for col in collections:
            lines.append(f"  - '{col.name}' → {col.count()} chunks stored")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to list collections: {str(e)}"


@tool
def delete_rag_collection(collection: str) -> str:
    """Permanently delete a RAG collection."""
    try:
        import chromadb
        safe_name = _safe_collection_name(collection)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        existing = [c.name for c in client.list_collections()]
        if safe_name not in existing:
            return f"Collection '{collection}' not found. Available: {', '.join(existing) or 'none'}"
        client.delete_collection(safe_name)
        logger.info(f"[RAG] Deleted collection '{safe_name}'")
        return f"Collection '{collection}' deleted successfully."
    except Exception as e:
        return f"Failed to delete collection: {str(e)}"
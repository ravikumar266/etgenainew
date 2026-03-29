"""
agent/tools_rag.py
──────────────────
RAG pipeline with cached ChromaDB client (Fix 5).

Fix 5: ChromaDB PersistentClient created ONCE at module load,
       reused across all requests — eliminates 2-5s init per call.
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

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# ── Fix 5: Single ChromaDB client — initialized once, reused forever ──────────
_chroma_client = None

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
            logger.info(f"[RAG] ChromaDB client initialized → {CHROMA_DIR}")
        except Exception as e:
            logger.error(f"[RAG] ChromaDB init failed: {e}")
            raise
    return _chroma_client


# ── Embeddings — lazy loaded ──────────────────────────────────────────────────
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        models = [
            "models/text-embedding-004",
            "models/embedding-001",
        ]
        for model_name in models:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                emb = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
                emb.embed_query("test")
                logger.info(f"[RAG] Embeddings: {model_name}")
                _embeddings = emb
                return _embeddings
            except Exception as e:
                logger.warning(f"[RAG] {model_name} failed: {e}")

    logger.warning("[RAG] No embeddings — using ChromaDB default")
    return None


# ── Text splitter ─────────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _get_vectorstore(collection: str) -> Chroma:
    """Use cached client — no re-initialization."""
    emb    = _get_embeddings()
    kwargs = dict(
        collection_name=_safe_collection_name(collection),
        persist_directory=CHROMA_DIR,
    )
    if emb is not None:
        kwargs["embedding_function"] = emb
    return Chroma(**kwargs)


def _safe_collection_name(name: str) -> str:
    safe = re.sub(r"[^a-z0-9\-]", "-", name.lower().strip())
    safe = re.sub(r"-+", "-", safe).strip("-")[:63]
    return (safe + "---")[:3] if len(safe) < 3 else safe


def _ingest_text(text: str, collection: str, source_label: str,
                 metadata_extra: Optional[dict] = None) -> str:
    chunks = _splitter.split_text(text)
    if not chunks:
        return f"No text extracted from '{source_label}'."
    metadata = [{"source": source_label, "chunk": i, **(metadata_extra or {})}
                for i in range(len(chunks))]
    vs = _get_vectorstore(collection)
    vs.add_texts(texts=chunks, metadatas=metadata)
    logger.info(f"[RAG] '{source_label}' → '{collection}' ({len(chunks)} chunks)")
    return f"✅ {len(chunks)} chunks from '{source_label}' into '{collection}'."


def _scrape_url(url: str) -> str:
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True)).strip()


def _load_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def _extract_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be",):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if vid else None
    if parsed.netloc in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            ids = parse_qs(parsed.query).get("v", [])
            return ids[0] if ids else None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
            return parts[1]
    return None


def _fetch_youtube_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        for finder in [
            lambda: transcript_list.find_manually_created_transcript(["en","en-US","en-GB"]),
            lambda: transcript_list.find_generated_transcript(["en","en-US","en-GB"]),
        ]:
            try: transcript = finder(); break
            except NoTranscriptFound: pass
        if not transcript:
            for t in transcript_list:
                transcript = t.translate("en") if t.language_code != "en" else t
                break
        if not transcript: return ""
        return " ".join(e["text"].strip() for e in transcript.fetch() if e.get("text","").strip())
    except Exception as e:
        logger.warning(f"[RAG] Transcript failed: {e}")
        return ""


def _fetch_youtube_metadata(video_id: str) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    result  = {"title":"","description":"","channel":"","published_at":"","tags":[],"duration":"","view_count":""}
    if not api_key: return result
    try:
        resp = requests.get(
            f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}",
            timeout=10
        )
        items = resp.json().get("items", [])
        if not items: return result
        s = items[0].get("snippet", {})
        result.update({
            "title": s.get("title",""), "description": s.get("description",""),
            "channel": s.get("channelTitle",""), "published_at": s.get("publishedAt",""),
            "tags": s.get("tags",[]),
            "duration": items[0].get("contentDetails",{}).get("duration",""),
            "view_count": items[0].get("statistics",{}).get("viewCount",""),
        })
    except Exception as e:
        logger.warning(f"[RAG] YouTube metadata failed: {e}")
    return result


@tool
def ingest_webpage(url: str, collection: str = "default") -> str:
    """Load a webpage into RAG. Args: url, collection (default: 'default')."""
    try:
        text = _scrape_url(url)
        if not text: return f"Page at {url} is empty."
        return _ingest_text(text, collection, url, {"type": "webpage"})
    except Exception as e:
        return f"Failed to ingest webpage: {str(e)}"


@tool
def ingest_pdf(file_path: str, collection: str = "default") -> str:
    """Load a PDF file into RAG. Args: file_path (absolute path), collection."""
    if not os.path.exists(file_path):
        return f"File not found: '{file_path}'"
    if not file_path.lower().endswith(".pdf"):
        return f"Not a PDF: '{file_path}'"
    try:
        text = _load_pdf(file_path)
        if not text: return f"No readable text in '{file_path}'."
        return _ingest_text(text, collection, os.path.basename(file_path),
                            {"type":"pdf","path":file_path})
    except Exception as e:
        return f"Failed to ingest PDF: {str(e)}"


@tool
def ingest_youtube(video_url: str, collection: str = "default") -> str:
    """Load YouTube transcript + metadata into RAG. Args: video_url, collection."""
    video_id = _extract_video_id(video_url)
    if not video_id: return f"Invalid YouTube URL: '{video_url}'"
    metadata   = _fetch_youtube_metadata(video_id)
    transcript = _fetch_youtube_transcript(video_id)
    title      = metadata.get("title") or f"YouTube {video_id}"
    lines = [f"Title: {title}", f"Channel: {metadata.get('channel','')}"]
    if metadata.get("description"): lines.append(f"Description:\n{metadata['description'][:2000]}")
    if transcript: lines.append(f"Transcript:\n{transcript}")
    full_text = "\n".join(lines)
    if not full_text.strip(): return f"No content for '{video_url}'"
    result = _ingest_text(full_text, collection, f"youtube:{video_id}",
                          {"type":"youtube","video_id":video_id,"title":title})
    words = len(transcript.split()) if transcript else 0
    return f"{result}\nTitle: {title}\nTranscript: {words} words"


@tool
def query_rag(question: str, collection: str = "default", top_k: int = 4) -> str:
    """Search RAG and return relevant chunks. Args: question, collection, top_k (1-10)."""
    try:
        vs    = _get_vectorstore(collection)
        count = vs._collection.count()
        if count == 0:
            return f"Collection '{collection}' is empty. Ingest content first."
        top_k   = min(max(1, top_k), 10)
        results = vs.similarity_search_with_relevance_scores(question, k=top_k)
        if not results:
            return f"No results in '{collection}' for: {question}"
        parts = [f"Retrieved {len(results)} chunks from '{collection}':\n"]
        for i, (doc, score) in enumerate(results, 1):
            src   = doc.metadata.get("source","unknown")
            dtype = doc.metadata.get("type","unknown").upper()
            parts.append(f"[{i}] {dtype} | {src} | {score:.0%}\n{doc.page_content.strip()}\n")
        return "\n".join(parts)
    except Exception as e:
        return f"RAG query failed: {str(e)}"


@tool
def list_rag_collections() -> str:
    """List all RAG collections and chunk counts. No input required."""
    try:
        client = _get_chroma_client()
        cols   = client.list_collections()
        if not cols: return "No RAG collections found."
        return "Collections:\n" + "\n".join(f"  - '{c.name}' → {c.count()} chunks" for c in cols)
    except Exception as e:
        return f"Failed to list collections: {str(e)}"


@tool
def delete_rag_collection(collection: str) -> str:
    """Delete a RAG collection permanently. Args: collection name."""
    try:
        client    = _get_chroma_client()
        safe_name = _safe_collection_name(collection)
        existing  = [c.name for c in client.list_collections()]
        if safe_name not in existing:
            return f"Collection '{collection}' not found. Available: {', '.join(existing) or 'none'}"
        client.delete_collection(safe_name)
        return f"Collection '{collection}' deleted."
    except Exception as e:
        return f"Failed to delete: {str(e)}"